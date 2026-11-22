"""Exploratory leave-one-cohort-out survival learning on fixed clinical inputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from numba import njit,prange,set_num_threads
from threadpoolctl import threadpool_limits

SCRIPT_DIR=Path(__file__).resolve().parent
OUT=SCRIPT_DIR/'eligible_sources_only'
EXT=SCRIPT_DIR.parents[1]
sys.path.insert(0,str(EXT/'scripts'))
from reference import TRAIN_RUN,get_cohorts,prepare_features,verify
from validate_external import weighted_c
SEED=20261121
EPOCHS=60
TARGETS=['nhanes3','charls','hrs','elsa','share','klosa','mimic_iv']
AUDIT={}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**22),b''): h.update(b)
    return h.hexdigest()

def read(p,**kw):
    AUDIT[str(p.relative_to(EXT.parent))]=digest(p)
    return pd.read_csv(p,**kw)

class Data:
    def __init__(self,d,bundle):
        d=d.loc[np.isfinite(d.time_yr)&d.time_yr.gt(0)&d.event.isin([0,1])&d.age.ge(20)&d.reindex(columns=bundle['features']).notna().any(axis=1)].copy()
        if 'ID' not in d:d=d.rename(columns={'SEQN':'ID'})
        assert d.ID.is_unique and d.ID.notna().all()
        d=d.sort_values('time_yr',ascending=False,kind='stable')
        self.n=len(d)
        self.t=d.time_yr.to_numpy()
        e=d.event.to_numpy(dtype='int64')
        x,m,age=prepare_features(d,bundle)
        self.e=torch.from_numpy(e.astype('float32'))
        self.x=torch.from_numpy(np.column_stack([age,x,m]).astype('float32'))
        self.age=torch.from_numpy(((d.age.to_numpy(dtype='float32')-60)/10))
        self.v1=d.full_lp.to_numpy() if 'full_lp' in d else None
        self.rankend=torch.from_numpy(np.searchsorted(-self.t,-self.t,side='right')-1)
        self.ids_digest=hashlib.sha256('\n'.join(d.ID.astype(str)).encode()).hexdigest()

def load_data():
    bundle=verify()
    dev,temporal,_=get_cohorts()
    a=read(TRAIN_RUN/'development_ids.csv.gz')
    pd.testing.assert_frame_equal(dev[['SEQN','cycle']],a,check_dtype=False)
    sets={'development':Data(dev.assign(time_yr=dev.permth_exm/12),bundle)}
    tmp=read(TRAIN_RUN/'temporal_predictions.csv.gz',usecols=['SEQN','full_lp'])
    sets['temporal']=Data(temporal.assign(time_yr=temporal.permth_exm/12).merge(tmp,on='SEQN',validate='one_to_one'),bundle)
    for c in TARGETS:
        if c=='mimic_iv':
            f=EXT/'mimic_iv'/'formal_v3_1'
            idx=read(f/'landmark_index.csv.gz',usecols=['subject_id','hadm_id','age','time_yr','event'])
            labs=read(f/'lab_panel.csv.gz')
            vt=read(f/'vitals.csv.gz',usecols=['subject_id','hadm_id','bmi','pulse','sbp','dbp'])
            raw=idx.merge(labs,on='hadm_id',validate='one_to_one').merge(vt,on=['subject_id','hadm_id'],how='left',validate='one_to_one').rename(columns={'subject_id':'ID'})
            pred=read(f/'predictions.csv.gz',usecols=['ID','full_lp','full_n_observed'])
        else:
            raw=read(EXT/'inputs'/f'{c}.csv.gz',dtype={'ID':str})
            pred=read(EXT/'results'/c/'predictions.csv.gz',usecols=['ID','full_lp','full_n_observed'],dtype={'ID':str})
        d=raw.merge(pred,on='ID',validate='one_to_one')
        assert np.array_equal(d.full_n_observed.to_numpy(),d.reindex(columns=bundle['features']).notna().sum(axis=1))
        sets[c]=Data(d,bundle)
    return sets,bundle

class SurvivalMLP(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(p,16),nn.Tanh(),nn.Linear(16,8),nn.Tanh(),nn.Linear(8,1,bias=False))
        self.age_slope=nn.Parameter(torch.tensor(.8))
    def forward(self,x,age):
        return self.age_slope*age+self.network(x).squeeze(1)

def fit(data,names):
    torch.manual_seed(SEED)
    model=SurvivalMLP(data[names[0]].x.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.015)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        for name in names:
            s=data[name]
            lp=model(s.x,s.age)
            # Breslow for tied event times. Each distinct cohort contributes one mean event loss.
            # A shift by a detached maximum keeps exp and cumulative sums well conditioned on CPU.
            shifted=lp-lp.detach().max()
            logrisk=shifted.exp().cumsum(0).log()[s.rankend]
            loss=((logrisk-shifted)*s.e).sum()/s.e.sum()/len(names)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
    return model

@torch.no_grad()
def rank(model,d):
    return model(d.x,d.age).numpy()

@njit(parallel=True)
def bootstrap(t,e,ranks):
    draws=np.empty((2000,ranks.shape[0]))
    for i in prange(2000):
        np.random.seed(SEED+i)
        w=np.bincount(np.random.randint(0,len(e),len(e)),minlength=len(e))
        for j in range(ranks.shape[0]):
            draws[i,j]=weighted_c(t,e,ranks[j],w)
    return draws

def checkpoint(m,path):
    loc=path.with_suffix('.tmp')
    torch.save(m.state_dict(),loc)
    loc.replace(path)

def main():
    torch.set_num_threads(3);set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    data,bundle=load_data()
    frozen_sha=digest(TRAIN_RUN/'model_freeze.json')
    spec={'model':'end-to-end mortality MLP (age skip + 16 tanh + 8 tanh + linear); no initialization from frozen v1',
          'features':bundle['features'],'age_input':'age skip; NHANES-standardized age in MLP',
          'mask':'43 binary observed markers appended to 43 NHANES-standardized marker inputs; no extra masking',
          'training':'AdamW lr .008, weight decay .015, 60 full batch epochs; mean cohort-specific Breslow loss (equal cohort weights); seed 20261121; no target tuning',
          'targets':TARGETS,'target_outcomes':'community all-cause mortality with cohort follow-up; MIMIC all-cause by day 365 from 24h landmark',
          'scope':'External datasets and their v1 results were already examined; exploratory. Two conventional NHANES cohorts are distinct survey cycles.',
          'development_only_ref':'Same newly initialized MLP trained on original development data alone',
          'uncertainty':'2000 paired patient bootstraps of held-out scores, conditional on fitted training sets; no survey or country clustering',
          'v1_freeze_sha256':frozen_sha,'sources':AUDIT,
          'cohort_rosters':{k:dict(n=d.n,deaths=int(d.e.sum().item()),roster_sha256=d.ids_digest) for k,d in data.items()}}
    config=OUT/'analysis_spec.json'
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if config.exists():
        assert json.loads(config.read_text(encoding='utf8'))==spec,'LOCO sources or protocol changed'
    else:config.write_text(json.dumps(spec,indent=2,ensure_ascii=False),encoding='utf8')
    def trainer(excluded,names):
        dest=OUT/f'mlp_{excluded}_{fingerprint[:12]}.pt'
        if dest.exists():
            m=SurvivalMLP(data[names[0]].x.shape[1])
            m.load_state_dict(torch.load(dest,map_location='cpu',weights_only=True))
        else:
            m=fit(data,names);checkpoint(m,dest)
        return m
    devmdl=trainer('development_only',['development'])
    metrics=[]
    for held in [*TARGETS,'temporal']:
        names=['development',*[c for c in TARGETS if c!=held]]
        assert held not in names and 'temporal' not in names
        out=OUT/f'{held}_{fingerprint[:12]}.json'
        if out.exists():
            metrics.append(json.loads(out.read_text(encoding='utf8')));continue
        m=trainer('exclude_'+held,names)
        d=data[held];r0=d.v1;r1=rank(devmdl,d);rm=rank(m,d)
        assert np.isfinite(np.column_stack([r0,r1,rm])).all()
        ranks=np.stack([np.unique(v,return_inverse=True)[1]+1 for v in [r0,r1,rm]])
        e=d.e.numpy().astype('int64')
        point=np.array([weighted_c(d.t,e,r,np.ones(d.n,dtype='int64')) for r in ranks])
        boots=bootstrap(d.t,e,ranks)
        assert np.isfinite(boots).all()
        val={'target':held,'fit_cohorts':','.join(names),'n':d.n,'events':int(e.sum()),'v1_C':point[0],'single_C':point[1],'multi_C':point[2]}
        for j,nm in [(0,'v1'),(1,'single')]:
            val['delta_multi_minus_'+nm]=point[2]-point[j]
            lo,hi=np.quantile(boots[:,2]-boots[:,j],[.025,.975])
            val['delta_vs_'+nm+'_lo']=lo;val['delta_vs_'+nm+'_hi']=hi
        temp=out.with_suffix('.tmp')
        temp.write_text(json.dumps(val,indent=2),encoding='utf8');temp.replace(out)
        metrics.append(val);print(held,d.n,point.tolist(),flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'leave_one_cohort_out.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=3):main()

