"""Independent pandas reconstruction on a deterministic patient sample; verify full-cohort evidence."""
import json
import numpy as np
import pandas as pd
from run import OUT,CACHE,TECH,MAP,DIFF,ITEMS,duckdb,compose_features,sha,write_json


def main():
    audit=json.loads((OUT/"verification.json").read_text(encoding="utf8"))
    for fname,digest in audit["results_sha256"].items():
        p=CACHE/fname if fname.endswith(".gz") else OUT/fname
        assert sha(p)==digest
    db=duckdb.connect(str(TECH/"icu_cache/time_limited.duckdb"),read_only=True)
    chosen=db.execute("select * from cohort order by subject_id").df().sample(150,random_state=5342)
    db.register("chosen",chosen)
    labs=db.execute("select x.*,c.intime,c.cutoff from hosp.labevents x join chosen c using(subject_id,hadm_id)").df()
    obs=db.execute("select x.*,c.intime,c.cutoff from icu.chartevents x join chosen c using(subject_id,hadm_id,stay_id)").df()
    db.close()
    selected=chosen.sort_values("subject_id").subject_id
    labs=labs[labs.itemid.isin(MAP)].copy()
    labs["feature"]=labs.itemid.map({i:v[0] for i,v in MAP.items()})
    expected=labs.itemid.map({i:v[2] for i,v in MAP.items()})
    unit=labs.valueuom.str.lower().str.strip().str.replace("µ","u").str.replace("μ","u")
    phys=labs.valuenum.gt(0)|(labs.itemid.isin(DIFF|{50889})&labs.valuenum.eq(0))
    accepted=unit.eq(expected)|(unit.eq("u/l")&expected.eq("iu/l"))|(unit.eq("mmol/l")&expected.eq("meq/l"))
    labs=labs[accepted&phys&np.isfinite(labs.valuenum)&(~labs.itemid.isin(DIFF)|labs.valuenum.le(100))].copy()
    labs["v"]=labs.valuenum.where(labs.itemid.ne(50889),labs.valuenum/10)
    obs=obs[obs.itemid.isin(ITEMS)].copy()
    unit=obs.valueuom.str.strip().str.lower()
    ok=unit.eq(obs.itemid.map({i:v[1] for i,v in ITEMS.items()}))
    ok&=obs.valuenum.between(obs.itemid.map({i:v[2] for i,v in ITEMS.items()}),obs.itemid.map({i:v[3] for i,v in ITEMS.items()}))
    obs=obs[ok]
    markers=[v[0] for v in MAP.values()]+["sbp","dbp","pulse","bmi"]
    for name,start in [("icu_minus6",6),("icu_strict",0)]:
        def within(v):
            return v.charttime.ge(v.intime-pd.Timedelta(hours=start))&v.charttime.lt(v.cutoff)&v.storetime.between(v.charttime,v.cutoff)
        ls=labs[within(labs)].sort_values(["charttime","storetime","labevent_id"]).drop_duplicates(["subject_id","feature"])
        wide=ls.pivot(index="subject_id",columns="feature",values="v")
        vit=compose_features(obs[within(obs)]).set_index("subject_id")
        reproduced=wide.join(vit[["sbp","dbp","pulse","bmi"]],how="outer").reindex(index=selected,columns=markers)
        trial=pd.read_csv(CACHE/(name+"_panel.csv.gz")).set_index("subject_id").reindex(index=selected,columns=markers)
        np.testing.assert_allclose(reproduced,trial,equal_nan=True)
    common=pd.read_csv(CACHE/"aligned_individual_scores.csv.gz").dropna(subset=["icu_minus6","icu_strict"])
    inc=pd.read_csv(OUT/"oof_increment.csv")
    main=inc[inc.population.eq("both_windows_with_lab")]
    assert len(main)==18 and main.q_BH_18_common.lt(.05).all()
    original=pd.read_csv(TECH/"icu_severity.csv")
    old=inc[(inc.population=="all_27005")&(inc.clock=="hospital24")].merge(original,left_on=["outcome","base_severity"],right_on=["endpoint","comparator"])
    np.testing.assert_allclose(old.incremental_auc_oof_x,old.incremental_auc_oof_y,atol=1e-10)
    write_json(OUT/"independent_check.json",dict(pandas_reconstruction_n=len(chosen),windows=2,marker_definitions=len(markers),
        equal_including_missing=True,common_matched_n=len(common),six_original_OOF_increments_reproduced=True,
        all_eighteen_common_patient_incremental_tests_at_FDR_05=True))


if __name__=="__main__":main()
