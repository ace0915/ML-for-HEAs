"""Regression: partial Al limits, process aliases, reviewable invalid plans, strict approval."""
import base64, time, io, zipfile
from unittest.mock import patch
import numpy as np, pandas as pd
from fastapi.testclient import TestClient
from app import app, sessions
client=TestClient(app)
els=['Fe','Ni','Co','Cr','Al']; rng=np.random.default_rng(45)
x=rng.dirichlet([3,3,3,3,5],64)*100
df=pd.DataFrame(x,columns=els); df['HV']=250+9*df.Al+2*df.Cr; df['Processing']=np.where(np.arange(len(df))%2,'As-cast','Annealed')
def upload(df):
 r=client.post('/api/upload',json={'name':'test.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()}); assert r.status_code==200,r.text; return r.json()
c=dict(elements=els,targets=['HV'],extras=[],categorical_extras=[],descriptors=[],selectors=['Pearson'],models=['Ridge'],raw=True,unit='at.%',group=None,k=3,budget=1,folds=2,test=.2,seed=42,optimizer='None')
settings={'HV':dict(models=['Ridge'],selectors=['Pearson'],k=3,optimizer='None',budget=1,folds=2,screen_top=1)}
def plan(u,design,config=c):
 with patch('app.request_plan',return_value={'rationale':'Mock','config':{'descriptors':[],'target_settings':settings},'design':design}):
  r=client.post('/api/plan/'+u['sid'],json={'config':config,'model':'mock','goal':'Al 35-50 at.%, maximize hardness','language':'zh'}); assert r.status_code==200,r.text; return r.json()
def body(u,p): return dict(config=p['config'],design=p['design'],signature=u['signature'],plan_id=p['plan_id'],approved=True)
u=upload(df); design=dict(bounds={'Al':[35,50]},process_mode='composition_only',directions={'HV':'max'},plot_axes={'x_element':'Al','y_target':'HV'},pool_size=40,iterations=8)
p=plan(u,design); assert not p['design_errors'],p; assert p['design']['bounds']['Al']==[35,50]; assert set(p['design']['bounds'])==set(els); assert p['design']['process_mode']=='fixed' and p['design']['conditions']=={}; assert p['design_sources']['bounds']=='LLM_partial+observed_ranges';print('Partial Al bounds + composition-only alias: PASS',flush=True)
p=plan(u,{**design,'bounds':{'Al':[50,35]}}); assert 'Al: invalid bounds' in p['design_errors'][0]; assert client.post('/api/workflow/'+u['sid'],json=body(u,p)).status_code==400; print('Reversed bounds remain reviewable, approval blocked: PASS',flush=True)
p=plan(u,{**design,'bounds':{e:[35,50] for e in els}}); assert 'sum(lower)' in p['design_errors'][0]; assert client.post('/api/workflow/'+u['sid'],json=body(u,p)).status_code==400; print('Infeasible sums remain reviewable, approval blocked: PASS',flush=True)
p=plan(u,{**design,'process_mode':'arbitrary_new_route'}); assert 'Unsupported process_mode' in p['design_errors'][0]; assert client.post('/api/workflow/'+u['sid'],json=body(u,p)).status_code==400;print('Unknown process mode not silently changed: PASS',flush=True)
p=plan(u,{**design,'bounds':{'Al':[35,50],'Zn':[0,5]}}); assert 'unknown=' in p['design_errors'][0]; print('Unknown element preserved for review: PASS',flush=True)
p=plan(u,{**design,'process_mode':'observed'}); assert not p['design_errors']; assert p['design']['process_mode']=='fixed';print('No process inputs + observed mode: PASS',flush=True)
q=plan(u,{**design,'process_mode':'observed combinations'}, {**c,'extras':['Processing'],'categorical_extras':['Processing']}); assert not q['design_errors'],q; assert q['design']['process_mode']=='observed' and not q['design']['conditions'];print('Observed process alias: PASS',flush=True)
q=plan(u,{**design,'process_mode':'fixed processing'}, {**c,'extras':['Processing'],'categorical_extras':['Processing']}); assert not q['design_errors']; assert q['design']['conditions']['Processing'] in ('As-cast','Annealed');print('Missing fixed processing value filled and documented: PASS',flush=True)
from regression_helpers import finish_stage2
p=finish_stage2(client,u['sid'],plan(u,design));r=client.post('/api/workflow/'+u['sid'],json=body(u,p));assert r.status_code==200,r.text
for i in range(600):
 status=client.get('/api/status/'+u['sid']).json()
 if status['status']!='running':break
 time.sleep(.1)
assert status['status']=='done',status;out=sessions[u['sid']]['design'];assert out.Al.between(35,50).all();assert np.allclose(out[els].sum(axis=1),100);assert 'envelope_single_objective' in status['inverse']['figures'];z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+u['sid']).content));assert 'figures/envelope_single_objective.png' in z.namelist();print('Approved end-to-end training, BO, Al range, plot and export: PASS',flush=True)
