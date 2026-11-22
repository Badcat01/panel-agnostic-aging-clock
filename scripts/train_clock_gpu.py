"""Train and freeze the NHANES mortality clock using development cycles only."""

import argparse
import hashlib
import itertools
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from clock_network import FEATS, GROUPS, LOGF, FTMask, augment_mask, breslow_loss
from nhanes_calibration import prepare_table

PROJECT = Path(__file__).resolve().parents[1]
CYCLES = (1999, 2001, 2003, 2005, 2007)
GRID = (32, 64, 128)
RATES = (0.0005, 0.001)
SEEDS = (0, 1, 2, 3, 4)
MICRO = 768
MODEL = dict(d=64, L=3, heads=4, drop=0.1)


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def get_cohorts():
    frame, dnam, meta = prepare_table(PROJECT)
    if not frame.SEQN.is_unique or frame.SEQN.isna().any() or not dnam:
        raise ValueError("SEQN missing/duplicated or DNAm holdout missing")
    if not frame.event.isin([0, 1]).all() or not (frame.permth_exm > 0).all():
        raise ValueError("Invalid survival endpoint")
    eligible = frame.loc[frame[FEATS].notna().sum(axis=1) * 2 >= len(FEATS)].copy()
    dev = eligible.loc[eligible.cycle.isin(CYCLES) & ~eligible.SEQN.isin(dnam)].reset_index(drop=True)
    temporal = eligible.loc[(eligible.cycle >= 2009) & ~eligible.SEQN.isin(dnam)].reset_index(drop=True)
    assert set(dev.cycle) == set(CYCLES) and not set(dev.SEQN) & (set(temporal.SEQN) | dnam)
    assert len(dev) + len(temporal) + int(eligible.SEQN.isin(dnam).sum()) == len(eligible)
    meta["cohort_counts"] = dict(source=int(len(frame)), eligible=int(len(eligible)), development=len(dev),
                                 temporal=len(temporal), eligible_dnam=int(eligible.SEQN.isin(dnam).sum()))
    return dev, temporal, meta


def arrays(df):
    x = df[FEATS].astype(float).copy()
    x[LOGF] = np.log(x[LOGF].clip(lower=0.001))
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Unexpected infinite input")
    return x.to_numpy(), df.age.to_numpy(dtype=float)


class Fold:
    def __init__(self, dev, fit_cycles):
        x, age = arrays(dev)
        self.fit = np.flatnonzero(dev.cycle.isin(fit_cycles))
        self.mu = np.nanmean(x[self.fit], axis=0)
        self.sd = np.nanstd(x[self.fit], axis=0, ddof=1)
        self.am = age[self.fit].mean()
        self.asd = age[self.fit].std(ddof=1)
        if not np.all(np.isfinite(self.mu) & (self.sd > 0)) or self.asd <= 0:
            raise ValueError("Invalid training-fold normalization")
        z = np.nan_to_num(np.clip((x-self.mu)/self.sd, -5, 5), nan=0)
        self.x = torch.as_tensor(z, dtype=torch.float32, device="cuda")
        self.m = torch.as_tensor(np.isfinite(x), device="cuda")
        self.a = torch.as_tensor((age-self.am)/self.asd, dtype=torch.float32, device="cuda")
        self.t = torch.as_tensor(dev.permth_exm.to_numpy(), device="cuda")
        self.e = torch.as_tensor(dev.event.to_numpy(), device="cuda")
        self.y = dev.cycle.to_numpy()
        self.dev = dev


@torch.inference_mode()
def infer(model, fold, rows, panel=None):
    model.eval()
    predictions = []
    for block in torch.as_tensor(rows, device="cuda").split(MICRO):
        mask = fold.m[block]
        if panel is not None:
            mask = mask & panel
        predictions.append(model(fold.x[block], mask, fold.a[block]).cpu().numpy())
    return np.concatenate(predictions)


def fit_model(fold, lr, seed, steps, val_cycles=()):
    torch.manual_seed(seed)
    model = FTMask(len(FEATS), **MODEL).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    idx = torch.as_tensor(fold.fit, device="cuda")
    x, mask, age, t, e = (a[idx] for a in (fold.x, fold.m, fold.a, fold.t, fold.e))
    blocks = list(range(0, len(idx), MICRO))
    scores_by_cycle = []
    started = time.monotonic()
    for step in range(1, steps+1):
        model.train()
        m = augment_mask(mask)
        states, preds = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in blocks:
                states.append(torch.cuda.get_rng_state())
                preds.append(model(x[i:i+MICRO], m[i:i+MICRO], age[i:i+MICRO]).float())
        # Evaluate the loss on every participant. Replay the same dropout draws
        # to propagate its exact score gradient without retaining all activations.
        lp = torch.cat(preds).detach().requires_grad_()
        loss = breslow_loss(lp.double(), t, e)
        grad, = torch.autograd.grad(loss, lp)
        opt.zero_grad(set_to_none=True)
        for j, i in enumerate(blocks):
            torch.cuda.set_rng_state(states[j])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay = model(x[i:i+MICRO], m[i:i+MICRO], age[i:i+MICRO]).float()
            if step == 1 and not torch.equal(replay.detach(), preds[j]):
                raise RuntimeError("Dropout replay did not reproduce forward scores")
            replay.backward(grad[i:i+MICRO])
        opt.step()
        if step in GRID or step == steps:
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            for year in val_cycles:
                ix = np.flatnonzero(fold.y == year)
                pred = infer(model, fold, ix)
                c = concordance_index(fold.dev.permth_exm.iloc[ix], -pred, fold.dev.event.iloc[ix])
                scores_by_cycle.append(dict(year=year, steps=step, lr=lr, c=float(c)))
            print(f"  step={step} loss={loss.item():.5f} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return model, scores_by_cycle


def choose(rows):
    table = pd.DataFrame(rows).groupby(["lr", "steps"], as_index=False).c.mean()
    best = table.sort_values(["c","steps","lr"], ascending=[False,True,True]).iloc[0]
    return float(best.lr), int(best.steps)


def cross_validate(dev, run):
    cache = run / "cv"
    cache.mkdir(exist_ok=True)
    inner, outer = {}, {}
    for size, store in [(2, inner), (1, outer)]:
        for omitted in itertools.combinations(CYCLES, size):
            fold = Fold(dev, sorted(set(CYCLES)-set(omitted)))
            rows = []
            for lr in RATES:
                key = "omit_" + "_".join(map(str,omitted)) + f"_lr{lr}.json"
                path = cache / key
                if not path.exists():
                    print(f"CV {key} n={len(fold.fit)}", flush=True)
                    _, scores = fit_model(fold, lr, 0, max(GRID), omitted)
                    save_json(path, scores)
                rows.extend(json.loads(path.read_text(encoding="utf8")))
            store[omitted] = rows
    nested = []
    for year in CYCLES:
        candidates = [r for key, rows in inner.items() if year in key for r in rows if r["year"] != year]
        lr, steps = choose(candidates)
        metric, = [r for r in outer[(year,)] if r["lr"] == lr and r["steps"] == steps]
        part = dev[dev.cycle == year]
        nested.append({**metric, "n":len(part), "deaths":int(part.event.sum())})
    lr, steps = choose([r for rows in outer.values() for r in rows])
    cv = dict(nested_outer=nested, nested_macro_c=float(np.mean([r["c"] for r in nested])),
              final_lr=lr, final_steps=steps, rule="mean cycle Harrell C; ties prefer fewer steps then lower lr")
    save_json(run/"cross_validation.json", cv)
    return cv


def calibrate(fold, models):
    result, expected = {}, {}
    dev = fold.dev
    def slope(column, vals, ix):
        z = dev.iloc[ix][["permth_exm", "event"]].copy()
        z[column] = vals
        return float(CoxPHFitter().fit(z, "permth_exm","event").params_[column])
    b_age = slope("age", dev.age.to_numpy(), fold.fit)
    for name, features in {"full":FEATS, **GROUPS}.items():
        panel = torch.as_tensor(np.isin(FEATS,features), device="cuda")
        ix = np.flatnonzero(dev[features].notna().any(axis=1))
        lp = np.mean([infer(model,fold,ix,panel) for model in models], axis=0).astype(float)
        scale = slope("lp", lp, ix) / b_age
        ag = dev.age.to_numpy()[ix]
        ba = ag.mean() + (lp-lp.mean())*scale
        intercept, coef = np.linalg.lstsq(np.c_[np.ones(len(ix)),ag], ba, rcond=None)[0]
        result[name] = dict(lp_mean=float(lp.mean()), scale_years=float(scale), age_mean=float(ag.mean()),
                            residual_intercept=float(intercept), residual_slope=float(coef), n=len(ix))
        expected[name] = pd.Series(ba,index=ix)
    return result, b_age, expected


def train(run):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA with BF16 support")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    run.mkdir(parents=True, exist_ok=True)
    bundle_dir = run / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    freeze = run / "model_freeze.json"
    if freeze.exists():
        recorded = json.loads(freeze.read_text(encoding="utf8"))["files_sha256"]
        if any(sha(bundle_dir / name) != digest for name, digest in recorded.items()):
            raise RuntimeError("Frozen bundle failed integrity verification")
        print("Existing frozen bundle verified", flush=True)
        return
    dev, _, metadata = get_cohorts()
    identity = dict(project_scripts={f:sha(PROJECT/"scripts"/f) for f in
                                     ("train_clock_gpu.py","clock_network.py","nhanes_calibration.py","score_clock.py")},
                    data={p.name:sha(p) for p in sorted((PROJECT/"data"/"processed").glob("*.gz"))},
                    gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                    numpy=np.__version__, pandas=pd.__version__, lifelines=lifelines.__version__,
                    python=platform.python_version(), seeds=SEEDS, lr_grid=RATES, steps_grid=GRID,
                    cycles=CYCLES, model=MODEL, microbatch=MICRO, masking=[0.4,0.35,0.25],
                    objective="full-cohort Breslow partial likelihood with exact tied risk sets; BF16 transformer, FP64 loss",
                    inner="leave two development cycles out; score each inner cycle separately", outer="leave one development cycle out",
                    train_ids_sha256=hashlib.sha256(dev.SEQN.to_numpy(dtype="<i8").tobytes()).hexdigest())
    marker = run/"protocol.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf8")) != json.loads(json.dumps(identity)):
        raise RuntimeError("Existing run identity differs; use a new output directory")
    save_json(marker, identity)
    save_json(run/"data_audit.json", metadata)
    dev[["SEQN","cycle"]].to_csv(run/"development_ids.csv.gz", index=False)
    print(f"GPU {identity['gpu']} n={len(dev)} deaths={dev.event.sum()}", flush=True)
    cv = cross_validate(dev, run)
    fold = Fold(dev, CYCLES)
    models, filenames = [], []
    for seed in SEEDS:
        name = f"ftmask_seed{seed}.pt"
        path = bundle_dir/name
        if not path.exists():
            print(f"FINAL seed={seed} lr={cv['final_lr']} steps={cv['final_steps']}", flush=True)
            model, _ = fit_model(fold, cv["final_lr"], seed, cv["final_steps"])
            tmp = path.with_suffix(".tmp")
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, tmp)
            tmp.replace(path)
        model = FTMask(len(FEATS), **MODEL).cuda().eval()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        if not all(torch.isfinite(t).all() for t in model.parameters()):
            raise FloatingPointError("Checkpoint contains nonfinite weights")
        models.append(model)
        filenames.append(name)
    calibration, b_age, expected = calibrate(fold, models)
    bundle = dict(features=FEATS, log_features=LOGF, groups=GROUPS, mean=fold.mu.tolist(), std=fold.sd.tolist(),
                  age_mean=float(fold.am), age_std=float(fold.asd), model_kwargs=MODEL, models=filenames,
                  calibration=calibration, age_cox_slope=b_age, n_train=len(dev), deaths=int(dev.event.sum()),
                  harmonization=metadata, unit_reference="02_build_dataset.py and nhanes_calibration.py")
    save_json(bundle_dir/"clock_bundle.json", bundle)
    from score_clock import score_frame
    examples = np.linspace(0,len(dev)-1,64,dtype=int)
    reloaded = score_frame(dev.iloc[examples], bundle_dir, "cuda")
    for panel, vals in expected.items():
        np.testing.assert_allclose(reloaded[f"{panel}_ba"].to_numpy(), vals.reindex(examples).to_numpy(),
                                   rtol=1e-5, atol=1e-3, equal_nan=True)
    for name in ("score_clock.py","clock_network.py","nhanes_calibration.py","02_build_dataset.py"):
        shutil.copyfile(PROJECT/"scripts"/name, bundle_dir/name)
    files = {p.name:sha(p) for p in sorted(bundle_dir.iterdir()) if p.is_file()}
    if len(set(files[n] for n in filenames)) != len(SEEDS):
        raise ValueError("Ensemble checkpoints are duplicates")
    save_json(run/"model_freeze.json", dict(utc=utc(), files_sha256=files, verified_rows=len(examples),
              panel_count=len(expected), peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    print("FROZEN bundle reloaded and verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    train(parser.parse_args().run.resolve())
