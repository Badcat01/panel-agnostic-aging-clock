"""Re-evaluate SAPS-II using only chronic-disease codes from finished MIMIC admissions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from study_paths import MIMIC_ARCHIVE
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
SUP = OUT.parents[1]
TECH = SUP / "technical"
MIM = SUP.parent / "mimic_iv"
sys.path[:0] = [str(MIM)]
import duckdb
import sqlglot
from mimic_iv_panels import compare_auc, ci95, oof_comparison

ICU = TECH / "icu_cache/sources"
HOSP = MIM / "local_3_1/hosp"
HISTORY = TECH / "prior_comorbidity/cache"
DAY = pd.Timedelta(days=1)
YEAR = pd.Timedelta(days=365)
KEY = ["subject_id", "hadm_id"]
CODE = SUP / "resources/mimic-code/mimic-iv/concepts"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def roster(admissions, stays):
    frame = pd.read_csv(
        HISTORY / "history_index_lab_panel.csv.gz",
        usecols=KEY + ["admittime", "dischtime", "landmark", "age", "sex", "hospital_event",
                       "event", "time_yr", "hospital_time_yr"],
        parse_dates=["admittime", "dischtime", "landmark"],
    )
    clocks = pd.read_csv(HISTORY / "all_scores.csv.gz", usecols=["ID", "hadm_id", "full_lp"])
    frame = frame.merge(clocks.rename(columns={"ID": "subject_id"}), on=KEY, validate="one_to_one")
    selected = frame.merge(stays, on=KEY, validate="one_to_many")
    selected = selected[selected.intime.ge(selected.admittime) & selected.intime.le(selected.landmark)]
    selected = selected.sort_values(["subject_id", "intime", "stay_id"]).drop_duplicates("subject_id")
    selected["cutoff"] = selected.intime + DAY
    lag = (selected.cutoff - selected.landmark) / YEAR
    selected = selected[
        selected.outtime.gt(selected.cutoff)
        & selected.hospital_time_yr.gt(lag)
        & selected.time_yr.gt(lag)
    ].copy()
    assert selected.subject_id.is_unique and selected.hadm_id.is_unique and selected.stay_id.is_unique
    assert selected.full_lp.notna().all()
    past = admissions[KEY + ["admittime", "dischtime", "hospital_expire_flag"]].merge(
        selected[KEY + ["admittime", "cutoff"]].rename(
            columns={"hadm_id": "index_hadm_id", "admittime": "index_admittime"}),
        on="subject_id", validate="many_to_one",
    )
    past = past[
        past.dischtime.lt(past.index_admittime)
        & past.dischtime.ge(past.admittime)
        & past.hospital_expire_flag.eq(0)
    ].copy()
    assert past.dischtime.lt(past.cutoff).all() and not past.hadm_id.isin(selected.hadm_id).any()
    assert past.subject_id.nunique() == len(selected)
    return selected, past


def dependency_order():
    reference = re.compile(r"physionet-data\.mimiciv_(derived|hosp|icu)\.(\w+)")
    paths = {p.stem: p for p in CODE.rglob("*.sql")}
    order = {}

    def visit(name):
        if name in order:
            return
        query = paths[name].read_text(encoding="utf8")
        for kind, dep in reference.findall(query):
            if kind == "derived":
                visit(dep)
        order[name] = query

    visit("sapsii")
    return order


def scan_sql(path, types=None):
    p = path.resolve().as_posix().replace("'", "''")
    suffix = f",types={types}" if types else ""
    return f"read_csv('{p}',header=true,sample_size=100000{suffix})"


def revised_sapsii(db, original_query):
    query = re.sub(r"`physionet-data\.mimiciv_(derived|hosp|icu)\.(\w+)`", r"\1.\2", original_query)
    return sqlglot.transpile(query, read="bigquery", write="duckdb")[0]


def build_scores(db, paths, admissions, stays, selected, past):
    for schema in ("hosp", "icu", "derived"):
        db.execute(f"CREATE SCHEMA {schema}")
    db.register("roster", selected)
    db.register("prior_map", past[["subject_id", "hadm_id", "index_hadm_id", "dischtime"]])
    db.register("adm", admissions)
    db.register("stays", stays)
    db.execute("CREATE TABLE hosp.admissions AS SELECT a.* FROM adm a SEMI JOIN roster c USING(subject_id,hadm_id)")
    db.execute("CREATE TABLE icu.icustays AS SELECT a.* FROM stays a SEMI JOIN roster c USING(subject_id,hadm_id,stay_id)")
    joins = {
        "patients": ("hosp", "v.subject_id=c.subject_id"),
        "services": ("hosp", "v.hadm_id=c.hadm_id AND v.transfertime<=c.cutoff"),
        "labevents": ("hosp", """v.subject_id=c.subject_id AND (v.hadm_id=c.hadm_id OR v.hadm_id IS NULL)
            AND v.charttime BETWEEN c.intime - INTERVAL '6 hours' AND c.cutoff
            AND v.storetime BETWEEN v.charttime AND c.cutoff"""),
        "chartevents": ("icu", """v.subject_id=c.subject_id AND v.hadm_id=c.hadm_id AND v.stay_id=c.stay_id
            AND v.charttime BETWEEN c.intime - INTERVAL '6 hours' AND c.cutoff
            AND v.storetime BETWEEN v.charttime AND c.cutoff"""),
        # MIMIC outputevents contains storetime, subject_id, hadm_id and stay_id.
        "outputevents": ("icu", """v.subject_id=c.subject_id AND v.hadm_id=c.hadm_id AND v.stay_id=c.stay_id
            AND v.charttime BETWEEN c.intime - INTERVAL '6 hours' AND c.cutoff
            AND v.storetime BETWEEN v.charttime AND c.cutoff"""),
    }
    for name, (schema, condition) in joins.items():
        types = {"value": "VARCHAR"} if name in ("labevents", "chartevents") else None
        print("Loading", name, flush=True)
        db.execute(f"CREATE TABLE {schema}.{name} AS SELECT v.* FROM {scan_sql(paths[name], types)} v JOIN roster c ON {condition}")
    print("Matching exclusively prior ICD histories", flush=True)
    db.execute(f"""
        CREATE TABLE prior_codes AS
        SELECT d.subject_id,p.hadm_id AS source_hadm_id,p.index_hadm_id,p.dischtime,d.icd_code,d.icd_version
        FROM {scan_sql(paths["diagnoses_icd"], {"icd_code": "VARCHAR"})} d
        JOIN prior_map p USING(subject_id,hadm_id)
    """)
    assert db.execute("""SELECT count(*) FROM prior_codes p JOIN roster r USING(subject_id)
        WHERE p.dischtime >= r.admittime OR p.source_hadm_id = r.hadm_id OR p.index_hadm_id <> r.hadm_id
    """).fetchone()[0] == 0
    history_counts = db.execute("SELECT count(*),count(DISTINCT subject_id) FROM prior_codes").fetchone()
    db.execute("""CREATE TABLE hosp.diagnoses_icd AS
        SELECT subject_id,index_hadm_id AS hadm_id,icd_code,icd_version FROM prior_codes WHERE false""")
    queries = dependency_order()
    for name, query in queries.items():
        print("Running concept", name, flush=True)
        db.execute(f"CREATE TABLE derived.{name} AS {revised_sapsii(db,query)}")
    before = db.execute("SELECT stay_id,sapsii AS sapsii_no_history FROM derived.sapsii").df()
    db.execute("""INSERT INTO hosp.diagnoses_icd
       SELECT subject_id,index_hadm_id,icd_code,icd_version FROM prior_codes""")
    after = db.execute(revised_sapsii(db, queries["sapsii"])).df()
    data = selected.merge(before, on="stay_id", validate="one_to_one").merge(
        after[["stay_id", "sapsii", "comorbidity_score"]], on="stay_id", validate="one_to_one",
    )
    assert len(data) == len(selected)
    np.testing.assert_allclose(data.sapsii - data.sapsii_no_history, data.comorbidity_score)
    assert set(data.comorbidity_score.unique()).issubset({0, 9, 10, 17})
    for name in ("labevents", "chartevents", "outputevents"):
        schema = joins[name][0]
        assert db.execute(f"""SELECT count(*) FROM {schema}.{name} v JOIN roster c USING(subject_id)
          WHERE NOT(v.storetime BETWEEN v.charttime AND c.cutoff) OR v.charttime>c.cutoff""").fetchone()[0] == 0
    return data, history_counts


def metrics(data):
    summaries = []
    nested = []
    names = ["sapsii_no_history", "sapsii", "full_lp"]
    for endpoint in ["hospital_event", "event"]:
        y = data[endpoint].to_numpy(dtype=bool)
        x = data[names].to_numpy(dtype=float)
        auc, covariance, _ = compare_auc(x, y)
        np.testing.assert_allclose(auc, [roc_auc_score(y, s) for s in x.T], atol=1e-12)
        for i, name in enumerate(names):
            lower, upper = ci95(auc[i], covariance[i, i], (0, 1))
            diff = auc[i] - auc[0]
            dlo, dhi = ci95(diff, covariance[i,i] + covariance[0,0] - 2*covariance[i,0], (-1,1))
            summaries.append(dict(
                outcome=endpoint, model=name, n=len(y), events=int(y.sum()), auc=auc[i], lower=lower,
                upper=upper, delta_vs_no_history=diff, delta_lower=dlo, delta_upper=dhi,
            ))
        for key in ["sapsii_no_history", "sapsii"]:
            paired = oof_comparison(data[[key]].to_numpy(), data.full_lp.to_numpy(), y, 20261122)
            nested.append(dict(outcome=endpoint, comorbidity=key, n=len(y), events=int(y.sum()), **paired))
    return pd.DataFrame(summaries), pd.DataFrame(nested)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    paths = {name: HOSP / f"{name}.csv.gz" for name in ("admissions", "patients", "labevents")}
    paths.update({name: ICU / f"{name}.csv.gz" for name in (
        "icustays", "services", "chartevents", "outputevents", "diagnoses_icd")})
    with zipfile.ZipFile(MIMIC_ARCHIVE) as z:
        lines = z.read("mimic-iv-3.1/SHA256SUMS.txt").decode().splitlines()
    manifest = {member.lstrip("*"): checksum for checksum, member in (line.split(maxsplit=1) for line in lines)}
    raw_sha = {}
    for name, path in paths.items():
        schema = "icu" if name in {"icustays","chartevents","outputevents"} else "hosp"
        raw_sha[name] = digest(path)
        assert raw_sha[name] == manifest[f"{schema}/{path.name}"], name
    reference_audit = json.loads((TECH / "prior_comorbidity/cohort_audit.json").read_text("utf8"))
    assert digest(HISTORY / "history_index_lab_panel.csv.gz") == reference_audit["panel_sha256"]
    comparison_audit = json.loads((TECH / "prior_comorbidity/verification.json").read_text("utf8"))
    assert digest(HISTORY / "all_scores.csv.gz") == comparison_audit["paired_score_sha256"]
    admissions = pd.read_csv(paths["admissions"], parse_dates=["admittime", "dischtime", "deathtime"])
    stays = pd.read_csv(paths["icustays"], parse_dates=["intime", "outtime"])
    selected, past = roster(admissions, stays)
    print("Eligible patients", len(selected), "previous admissions", len(past), flush=True)
    with duckdb.connect() as db:
        db.execute(f"SET threads={args.threads}")
        db.execute("SET memory_limit='6GB'")
        db.execute("SET preserve_insertion_order=false")
        data, history_counts = build_scores(db, paths, admissions, stays, selected, past)
    main_auc, increment = metrics(data)
    main_auc.to_csv(OUT / "sapsii_prior_same_people_auc.csv", index=False)
    increment.to_csv(OUT / "sapsii_prior_same_people_increment.csv", index=False)
    audit = dict(
        n=len(data), hospital_deaths=int(data.hospital_event.sum()), one_year_deaths=int(data.event.sum()),
        n_prior_hospitalizations=len(past), prior_ICD_rows=history_counts[0], n_with_prior_ICD=history_counts[1],
        n_by_comorbidity_points={int(k): int(v) for k,v in data.comorbidity_score.value_counts().items()},
        no_current_admission_ICDs=True, prior_admissions_finished_before_index_admission=True,
        timing_checks_passed=True, independent_sklearn_check_passed=True, sum_of_scores_check_passed=True,
        source_sha256=raw_sha, reference_cohort_sha256=reference_audit["panel_sha256"],
        score_cache_sha256=digest(HISTORY / "all_scores.csv.gz"),
        source_SAPS_II_SQL_sha256=digest(CODE / "score/sapsii.sql"),
    )
    (OUT / "sapsii_prior_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(main_auc.to_string(index=False))
    print(increment.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
