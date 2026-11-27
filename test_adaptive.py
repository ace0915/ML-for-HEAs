"""Regression tests for adaptive planning and actual target-specific execution."""
import base64,json,time,io,zipfile
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app,sessions
from engine import split_target
import numpy as np,pandas as pd
client=TestClient(app);rng=np.random.default_rng(42);els=['Fe','Ni','Co','Cr','Al']
def data(n):
 x=rng.dirichlet([4]*5,n)*100;df=pd.DataFrame(x,columns=els);df['CR(%)']=np.arange(n)%3*20;df['YS']=250+8*x[:,4]+4*df['CR(%)'];df['EL']=30-.2*x[:,3];return df
c=dict(elements=els,targets=['YS','EL'],extras=['CR(%)'],categorical_extras=[],descriptors=[],selectors=[],models=['Decision tree'],raw=True,unit='at.%',group=None,k=1,budget=1,folds=2,test=.4,seed=42,optimizer='None')
def upload(df):
 r=client.post('/api/upload',json={'name':'synthetic.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()});assert r.status_code==200,r.text;return r.json()
def plan(u,response):
 with patch('app.request_plan',return_value=response):
  r=client.post('/api/plan/'+u['sid'],json={'config':c,'model':'mock','goal':'strength and elongation','language':'zh'});assert r.status_code==200,r.text;return r.json()
small=data(48);large=data(600);u=upload(small);v=upload(large)
partial={'rationale':'Mock omitted settings','config':{},'design':None};a=plan(u,partial);b=plan(v,partial)
assert a['config']['target_settings']['YS']['models']!=b['config']['target_settings']['YS']['models'];assert a['config']['target_settings']['YS']['budget']!=b['config']['target_settings']['YS']['budget'];assert a['config']['models']!=c['models'];assert a['setting_sources']['YS']['models']=='adaptive_data_rule';print('Small and large datasets generate different models and budgets without UI-default fallback',flush=True)
idx,tr,te,cv=split_target(small,'YS',None,a['config']['test'],42,a['config']['folds']);changed=small.copy();changed.loc[idx[te],'YS']+=100000;z=plan(upload(changed),partial);assert z['training_profiles']['YS']==a['training_profiles']['YS'];print('Holdout target perturbation does not change target training diagnostics',flush=True)
settings={'YS':dict(models=['Ridge','Extra trees'],selectors=['Pearson'],k=4,optimizer='Simulated annealing',budget=3,folds=3,screen_top=1),'EL':dict(models=['Ridge','Bayesian ridge','Huber'],selectors=['Spearman'],k=5,optimizer='None',budget=1,folds=2,screen_top=3)}
p=plan(u,{'rationale':'Mock explicit target settings','config':{'target_settings':settings,'descriptors':['S_mix_J_molK']},'design':{'directions':{'YS':'max','EL':'max'},'pool_size':70,'iterations':10}});assert p['config']['target_settings']==settings;assert all(q=='LLM_target' for q in p['setting_sources']['YS'].values());print('Explicit target-specific LLM settings applied exactly',flush=True)
from regression_helpers import finish_stage2
p=finish_stage2(client,u['sid'],p)
body=dict(config=p['config'],design=p['design'],signature=u['signature'],plan_id=p['plan_id'],approved=True)
wrong=json.loads(json.dumps(body));wrong['config']['test']=.3;assert client.post('/api/workflow/'+u['sid'],json=wrong).status_code==400
r=client.post('/api/workflow/'+u['sid'],json=body);assert r.status_code==200,r.text
for i in range(600):
 d=client.get('/api/status/'+u['sid']).json()
 if d['status']!='running':break
 time.sleep(.1)
assert d['status']=='done',d;assert d['summary']['target_settings']==settings;comp=pd.DataFrame(d['summary']['comparison']);assert set(comp[(comp.target=='YS')&(comp.stage=='screen')].model)==set(settings['YS']['models']);assert sum((comp.target=='YS')&(comp.stage=='final'))==1;assert sum((comp.target=='EL')&(comp.stage=='final'))==3;assert len(d['summary']['metrics'])==2;assert d['inverse']['evaluated']==d['inverse']['pool'] and d['inverse']['bo_evaluated']==10
s=sessions[u['sid']];assert 0<len(s['traces'])<=3;assert all(s['traces'].target=='YS');assert all(s['results'][t]['model'].named_steps['select'].k==settings[t]['k'] for t in settings)
z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+u['sid']).content));prov=json.loads(z.read('provenance.json'));assert prov['approved_workflow']['config']['target_settings']==settings;print('Approved execution: distinct models, selectors, budgets, CV screening, Pareto and export PASS',flush=True)
