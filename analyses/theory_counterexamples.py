"""Exact finite-state witnesses for overlap covariance sign and limits of D.
Each witness enumerates every independent bit-flip state (not Monte Carlo).
"""
import itertools
import numpy as np
from pathlib import Path
import csv

OUT=Path(__file__).resolve().parents[1]/'results/derived/theory_counterexamples_recomputed.csv'; OUT.parent.mkdir(parents=True,exist_ok=True)

def moments(labels, neighborhoods,p=0.1):
    y=np.array(labels,dtype=int); N=np.asarray(neighborhoods,dtype=int)
    base=(y[N].sum(axis=1) > (N.shape[1]/2)).astype(int)
    D=[]
    for i in range(len(y)):
        z=y.copy();z[i]^=1
        D.append(((z[N].sum(axis=1) > N.shape[1]/2).astype(int)!=base).astype(int))
    D=np.array(D).T
    probs=[]; obs=[]
    for bit in itertools.product([0,1], repeat=len(y)):
        arr=np.array(bit)
        w=np.prod(np.where(arr,p,1-p)); ny=y^arr
        changes=((ny[N].sum(axis=1)>N.shape[1]/2).astype(int)!=base).astype(int)
        probs.append(w);obs.append(changes)
    probs=np.asarray(probs); obs=np.asarray(obs)
    E=probs @ obs; joint=obs.T @ (obs * probs[:,None]); cov=joint-np.outer(E,E)
    Rvar=np.sum(cov)/(len(N)**2)
    V_ind=np.sum(E*(1-E))/(len(N)**2)
    return {'E':E,'cov':cov,'D':D,'exactvar':Rvar,'indvar':V_ind}

cases={
    'identical_neighbors': ([0,0,1,1], [[0,1,2],[0,1,2]]),
    'two_decisive_neighbors_shared': ([0,0,1,1],[[0,1,2],[0,1,3]]),
    'opposite_margin_shared_zero': ([0,0,1,1,1],[[0,1,2],[0,3,4]]),
}
rows=[]
for name,case in cases.items():
    z=moments(*case)
    rows.append({'case':name,'n_bits':len(case[0]), 'k':len(case[1][0]),'q':len(case[1]),
        'p':0.1,'pi_0':z['E'][0],'pi_1':z['E'][1],'covariance':z['cov'][0,1],
        'exact_variance':z['exactvar'],'query_independent_variance':z['indvar'],
        'D_rows':repr(z['D'].tolist())})
assert np.array_equal(moments(*cases['identical_neighbors'])['D'], moments(*cases['two_decisive_neighbors_shared'])['D'])
assert rows[2]['covariance'] < 0
with OUT.open('w',newline='') as o:
 w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:print(r)
