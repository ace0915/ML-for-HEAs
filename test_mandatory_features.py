"""V4.8 regression: mandatory ranking, approval Top-K and full-pool inverse."""
import base64,copy,io,time,zipfile
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app,sessions
from engine import SelectFeatures, SELECTORS, split_target
import numpy as np,pandas as pd
rng=np.random.default_rng(12); x=rng.normal(size=(72,5)); y=5*x[:,0]+.2*x[:,1]+rng.normal(size=72)*.1
try:
    SelectFeatures((),3).fit(x,y)
    raise AssertionError('empty methods accepted')
except ValueError: pass
for method in SELECTORS:
    sel=SelectFeatures((method,),3).fit(x,y)
    assert len(sel.indices_)==3 and np.isfinite(sel.scores_).all() and not np.all(sel.scores_==1),method
for method in ('RFE (Ridge)','Sequential forward (Ridge)'):
    sel=SelectFeatures((method,),5).fit(x,y)
    assert len(sel.indices_)==5 and np.ptp(sel.scores_)>0,method
    assert SelectFeatures((method,),1).fit(x[:,:1],y).indices_.tolist()==[0]
assert np.allclose(SelectFeatures(('Pearson',),3).fit(np.ones((72,5)),y).scores_,.6)
print('10 selectors, no empty bypass, RFE/SFS, genuine ties PASS',flush=True)
client=TestClient(app); els=['Fe','Ni','Co','Cr','Al']; a=rng.dirichlet([3]*5,72)*100; df=pd.DataFrame(a,columns=els);df['Processing']=np.where(np.arange(len(df))%2,'Annealed','As-cast');df['YS']=200+8*a[:,4]+np.arange(len(df))%2*40;df['EL']=35-.5*a[:,3]
c=dict(elements=els,targets=['YS','EL'],extras=['Processing'],categorical_extras=['Processing'],descriptors=[],selectors=['Pearson','Extra trees importance'],models=['Ridge'],raw=True,unit='at.%',group=None,k=3,budget=1,folds=2,test=.2,seed=42,optimizer='None')
def upload(d):
    r=client.post('/api/upload',json=dict(name='synthetic.csv',data=base64.b64encode(d.to_csv(index=False).encode()).decode()));assert r.status_code==200,r.text;return r.json()
u=upload(df);sid=u['sid']
def preview(c,sid=sid):
    r=client.post('/api/feature-preview/'+sid,json=c);assert r.status_code==200,r.text;return r.json()
assert client.post('/api/train/'+sid,json=dict(c,selectors=[])).status_code==400
p=preview(c);assert all(v['effective_k']==3 and sum(r['selected'] for r in v['ranking'])==3 and not v['tied'] for v in p.values())
for t in c['targets']:
    idx,tr,te,cv=split_target(df,t,None,.2,42,2);changed=df.copy();changed.loc[idx[te],t]+=1e6;assert preview(c,upload(changed)['sid'])[t]==p[t]
print('Manual preview and holdout perturbation isolation PASS',flush=True)
r=client.post('/api/adaptive-plan/'+sid,json={'config':c,'language':'zh'});assert r.status_code==200,r.text;assert all(v['selectors'] for v in r.json()['config']['target_settings'].values())
settings={t:dict(models=['Ridge'],selectors=[],k=3,optimizer='None',budget=1,folds=2,screen_top=1) for t in c['targets']}
with patch('app.request_plan',return_value={'rationale':'mock','config':{'target_settings':settings,'descriptors':[]},'design':{'directions':{'YS':'max','EL':'max'},'pool_size':35,'iterations':5}}):
    r=client.post('/api/plan/'+sid,json={'config':c,'model':'mock','language':'zh'});assert r.status_code==200,r.text;plan=r.json()
assert all(v['selectors'] for v in plan['config']['target_settings'].values());assert plan['setting_sources']['YS']['selectors']=='mandatory_adaptive_rule'
config=copy.deepcopy(plan['config']);config['target_settings']['YS'].update(k=2,selectors=['Pearson']);config['target_settings']['EL'].update(k=4,selectors=['Spearman']);config['k']=4;config['selectors']=['Pearson','Spearman'];p=preview(config)
from regression_helpers import finish_stage2
plan=finish_stage2(client,sid,plan,config=config)
body=dict(config=config,design=plan['design'],approved=True,signature=u['signature'],plan_id=plan['plan_id']);bad=copy.deepcopy(body);bad['config']['target_settings']['YS']['selectors']=[];assert client.post('/api/workflow/'+sid,json=bad).status_code==400
r=client.post('/api/workflow/'+sid,json=body);assert r.status_code==200,r.text
for i in range(500):
    result=client.get('/api/status/'+sid).json()
    if result['status']!='running':break
    time.sleep(.1)
assert result['status']=='done',result
for t,k in [('YS',2),('EL',4)]:
    q=sessions[sid]['results'][t];sel=q['model'].named_steps['select'];assert len(sel.indices_)==k;assert q['ranking']['selected'].sum()==k;assert q['ranking'].feature.tolist()==[r['feature'] for r in p[t]['ranking']];assert np.allclose(q['ranking'].score,[r['score'] for r in p[t]['ranking']]);assert q['model'].named_steps['model'].n_features_in_==k
assert result['inverse']['pool']==result['inverse']['evaluated']
z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+sid).content));assert 'rank_score: Pearson' in z.read('models/1_ranking.csv').decode();print('LLM fallback, user K=2/4, actual model inputs, full-pool Pareto, CSV export PASS',flush=True)
