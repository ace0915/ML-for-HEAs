"""Full-pool regression: controlled deterministic property surrogates."""
from unittest.mock import patch
import numpy as np
import pandas as pd
from engine import inverse, pool_grid

els=['Al','Fe']; bounds={'Al':[35,50],'Fe':[50,65]}; pool=pool_grid(els,bounds,1,100,42)
assert len(pool)==16
# Pick an optimum that is definitely outside the initial 3 BO selections.
bo=np.random.default_rng(42).choice(len(pool),3,replace=False)
missing=next(i for i in range(len(pool)) if i not in bo)
peak=float(pool.iloc[missing].Al)
class Model:
    def __init__(self,fn): self.fn=fn; self.calls=[]
    def predict(self,x):
        a=x.Al.to_numpy()  # raw composition is in at.%
        self.calls.extend(a.tolist()); return self.fn(a)

def run(directions,thresholds=None,iterations=3):
    m1=Model(lambda a:1000-(a-peak)**2)
    m2=Model(lambda a:np.abs(a-peak))
    res={'HV':{'model':m1,'cols':['Al']},'Cost':{'model':m2,'cols':['Al']}}
    out,h,n=inverse(pool,res,els,[],[],True,bounds,{},directions,step=1,pool_size=100,iterations=iterations,seed=42,thresholds=thresholds)
    assert n==len(out)==16 and out.pool_index.is_unique
    assert len(h)==min(16,iterations)
    assert len(m1.calls)==16 and len(set(np.round(m1.calls,8)))==16
    if 'Cost' in directions: assert len(m2.calls)==16
    assert out.BO_iteration.notna().sum()==len(h)
    assert out.loc[out.BO_iteration.notna()].sort_values('BO_iteration').pool_index.tolist()==h.pool_index.tolist()
    assert (out.evaluation_stage=='BO').sum()==len(h)
    assert np.allclose(out.pred_HV,1000-(out.Al-peak)**2)
    return out

out=run({'HV':'max','Cost':'min'})
assert out.Pareto.sum()==1 and out.loc[out.Pareto,'pool_index'].iloc[0]==missing
assert out.loc[out.Pareto,'evaluation_stage'].iloc[0]=='full_pool'
print('Optimum outside BO sample discovered; mixed max/min; all predictions exactly once; history aligned: PASS',flush=True)
out=run({'HV':'max','Cost':'max'})
assert out.Pareto.all()  # exact ties and trade-offs are non-dominated
print('Multiple non-dominated solutions and tied objectives: PASS',flush=True)
out=run({'HV':'max','Cost':'min'},{'HV':1001})
assert not out.Feasible.any() and not out.Pareto.any()
print('No feasible performance solution: PASS',flush=True)
out=run({'HV':'max','Cost':'min'},{'Cost':0})
assert out.Feasible.sum()==1 and out.Pareto.sum()==1
print('Minimized-objective upper threshold: PASS',flush=True)
out=run({'HV':'max'},iterations=16)
assert out.BO_iteration.notna().all() and out.Pareto.sum()==1
print('BO covers pool, no duplicate prediction: PASS',flush=True)
