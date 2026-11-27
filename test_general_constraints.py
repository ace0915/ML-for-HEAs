"""Regression tests for generic alloy/process constraints; synthetic data only."""
import base64, io, zipfile, time
from unittest.mock import patch
import numpy as np, pandas as pd
from fastapi.testclient import TestClient
from app import app, sessions
client=TestClient(app); rng=np.random.default_rng(51)
els=['Ti','Zr','Nb','Ta','Mo']; n=72
x=rng.dirichlet([5]*5,n)*100; df=pd.DataFrame(x,columns=els)
df['Temperature']=np.tile([700,800,900,1000],n//4);df['HoldingTime']=np.tile([1.,2.,3.],n//3);df['Processing']=np.where(np.arange(n)%2,'Annealed','Forged');df['YS']=400+9*df.Ti+2*df.Nb+.1*df.Temperature;df['EL']=40-.4*df.Ti+.01*df.Temperature
u=client.post('/api/upload',json={'name':'refractory.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()}).json(); sid=u['sid']
c=dict(elements=els,targets=['YS','EL'],extras=['Temperature','HoldingTime','Processing'],categorical_extras=['Processing'],descriptors=[],selectors=['Pearson'],models=['Ridge'],raw=True,unit='at.%',group=None,k=3,budget=1,folds=2,test=.2,seed=42,optimizer='None')
settings={t:dict(models=['Ridge'],selectors=['Pearson'],k=3,optimizer='None',budget=1,folds=2,screen_top=1) for t in c['targets']}
def plan(d):
 with patch('app.request_plan',return_value={'rationale':'Mock','config':{'descriptors':[],'target_settings':settings},'design':d}):
  r=client.post('/api/plan/'+sid,json={'config':c,'model':'mock','goal':'Synthetic refractory alloy design'});assert r.status_code==200,r.text;return r.json()
def check(d):return client.post('/api/validate-design/'+sid,json={'config':c,'design':d})
def wait():
 for _ in range(600):
  st=client.get('/api/status/'+sid).json()
  if st['status']!='running':assert st['status']=='done',st;return st
  time.sleep(.1)
 raise RuntimeError('timeout')
d=dict(bounds={'钛':{'min':'15 at.%','max':'30 at.%'},'TA':12.5,'Nb(at%)':'10–25 at.%'},process_mode='joint optimization',process_bounds={'temperature':{'min':'825','max':'875'},'HoldingTime':[1.5,2.5]},conditions={'Processing':['Annealed']},directions={'YS':'maximize','EL':'max'},pool_size=40,iterations=10)
p=plan(d);assert not p['design_errors'],p['design_errors']; q=p['design']; assert q['bounds']['Ti']==[15,30] and q['bounds']['Ta']==[12.5,12.5] and q['bounds']['Nb']==[10,25];assert q['step']==.5 and q['process_mode']=='bounded';assert q['process_ranges']=={'Temperature':[825,875],'HoldingTime':[1.5,2.5]};assert q['process_categories']=={'Processing':['Annealed']};print('Generic refractory system, alias/scalar/range parsing and automatic finer omitted grid: PASS',flush=True)
for label,patchd,expected in [('reversed element',{'bounds':{'Ti':[30,15]}},'Ti: invalid bounds'),('unknown element',{'bounds':{'W':[0,10]}},'unknown='),('explicit incompatible grid',{'step':1},'grid'),('unknown process',{'process_ranges':{'Invented':[1,2]}},'unknown or unselected'),('reversed process',{'process_ranges':{'Temperature':[900,800]}},'invalid process range'),('outside domain',{'process_ranges':{'Temperature':[650,875]}},'outside observed domain'),('bad category',{'process_categories':{'Processing':['Unknown']}},'recorded categories'),('conflicting field',{'conditions':{'Temperature':850}},'Conflicting'),('observed empty',{'process_mode':'observed'},'No complete recorded')]:
 bad={**q,**patchd};r=check(bad);assert r.status_code==400,(label,r.text);assert expected in r.json()['detail'],(label,r.text);print(label+': precise validation PASS',flush=True)
r=check(q);assert r.status_code==200,r.text
from regression_helpers import finish_stage2
p=finish_stage2(client,sid,p,design=q)
r=client.post('/api/workflow/'+sid,json={'approved':True,'signature':u['signature'],'plan_id':p['plan_id'],'config':p['config'],'design':q});assert r.status_code==200,r.text;st=wait();out=pd.DataFrame(st['inverse']['rows']);assert out.Ti.between(15,30).all() and out.Ta.eq(12.5).all() and out.Nb.between(10,25).all();assert np.allclose(out[els].sum(axis=1),100);assert out.Temperature.between(825,875).all() and out.HoldingTime.between(1.5,2.5).all() and out.Processing.eq('Annealed').all();assert not out.Temperature.isin(df.Temperature).any();print('Approved bounded joint BO, two ranges, category, fixed element, total and two targets: PASS',flush=True)
obs={**q,'process_mode':'observed','process_ranges':{'Temperature':[800,1000],'HoldingTime':[1,2]},'process_categories':{'Processing':['Annealed']}};r=client.post('/api/inverse/'+sid,json=obs);assert r.status_code==200,r.text;st=wait();out=pd.DataFrame(st['inverse']['rows']);assert out.Temperature.between(800,1000).all() and out.HoldingTime.between(1,2).all() and out.Processing.eq('Annealed').all();assert set(map(tuple,out[c['extras']].values))<=set(map(tuple,df[c['extras']].values));print('Manual constrained observed tuple BO: PASS',flush=True)
z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+sid).content));assert any('design' in f for f in z.namelist());assert st['inverse']['process_ranges']==obs['process_ranges'];print('Constraint provenance/export: PASS',flush=True)
