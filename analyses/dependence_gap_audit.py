"""Narrow audit reusing E3's published neighborhoods/scenarios without new datasets.
Independent approximate batch variance uses the exact per-query change probabilities
but discards pair covariances. This is a sensitivity analysis, not a deployment decision test.
"""
from pathlib import Path
import importlib.util
import sys
import os
import csv
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT
SAVED=ROOT/'results/processed/e3_probability_risk.csv'
NEW=ROOT/'results/derived/dependence_gap_recomputed.csv'
ENV=Path(os.environ.get('KNN_DATA_DIR',str(ROOT/'data/processed')))
sys.path.insert(0,str(SOURCE/'src'))
spec=importlib.util.spec_from_file_location('run_revision_experiments',SOURCE/'experiments/run_revision_experiments.py')
runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
runner.DATA_DIR=ENV
from knn_reliability.knn import build_neighbor_cache
from knn_reliability.probability import batch_risk_moments

src=pd.read_csv(SAVED);out=[]
for dataset in ['breast_cancer_wisconsin','digits_0_vs_8','imbalanced_binary','noisy_binary']:
    x,y=runner.load_dataset(dataset,0,max_samples=800)
    xtr,ytr,_,_,xt,_=runner.split_and_scale(x,y,0)
    classes=np.unique(ytr)
    for (numq,k),rows in src[src.dataset==dataset].groupby(['queries','k'],sort=True):
        neigh=build_neighbor_cache(xtr,xt[:int(numq)],int(k)).indices
        for _,r in rows.iterrows():
            scenario=r.probability_scenario
            if scenario=='heterogeneous_rank': p=np.linspace(.005,.20,len(ytr))
            elif scenario=='high_uniform': p=np.full(len(ytr),.15)
            elif scenario=='low_uniform': p=np.full(len(ytr),.01)
            else: p=np.full(len(ytr),.05)
            moments=batch_risk_moments(ytr,neigh,p,classes=classes)
            pi=moments.query_probabilities
            independent=float(np.sum(pi*(1-pi))/len(pi)**2)
            exact=float(moments.variance)
            released=float(r.variance_exact_shared_flips)
            out.append(dict(dataset=dataset,query_count=int(numq),k=int(k),scenario=scenario,
                            assumption=r.assumption,overlap_fraction=float(r.overlap_pair_fraction),
                            exact_variance=exact,query_independence_variance=independent,
                            dependence_correction=exact-independent,
                            signed_relative_correction=(exact-independent)/exact if exact>1e-14 else np.nan,
                            absolute_relative_error=abs(exact-independent)/exact if exact>1e-14 else np.nan,
                            released_exact_variance=released,absolute_reproduction_delta=abs(released-exact),
                            n_overlap_edges=int(moments.overlap_pair_count)))
    print('complete',dataset,flush=True)
df=pd.DataFrame(out);NEW.parent.mkdir(parents=True, exist_ok=True);df.to_csv(NEW,index=False,float_format='%.16g')
print('saved',len(df),NEW)
print('worst reproduction',df.absolute_reproduction_delta.max());assert df.absolute_reproduction_delta.max()<1e-11
a=df[df.assumption=='independent_flips'];assert len(df)==84 and len(a)==68
print('valid model rows',len(a),'datasets',a.dataset.nunique(),'positive',int((a.dependence_correction>1e-12).sum()),'negative',int((a.dependence_correction< -1e-12).sum()),'zero',int((a.dependence_correction.abs()<=1e-12).sum()))
print('abs_rel_err median p90 max',a.absolute_relative_error.median(),a.absolute_relative_error.quantile(.9),a.absolute_relative_error.max())
print('dataset medians',a.groupby('dataset')['absolute_relative_error'].median().to_dict())
print('top signed pos/neg')
print(a.nlargest(4,'dependence_correction')[['dataset','k','scenario','dependence_correction','absolute_relative_error']].to_string(index=False))
print(a.nsmallest(4,'dependence_correction')[['dataset','k','scenario','dependence_correction','absolute_relative_error']].to_string(index=False))
