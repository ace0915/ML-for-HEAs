"""Synthetic integration and mocked provider protocol tests, no credentials required."""
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from app import app, sessions
from llm import request_plan
import base64, json, time, io, zipfile
import numpy as np, pandas as pd
from engine import pareto
client=TestClient(app); rng=np.random.default_rng(42); els=['Fe','Ni','Co','Cr','Al']; x=rng.dirichlet([3]*5,70)*100; df=pd.DataFrame(x,columns=els); df['YS']=300+10*x[:,4]+2*x[:,3]; df['EL']=40-.5*x[:,4]
c=dict(elements=els,targets=['YS','EL'],extras=[],descriptors=['S_mix_J_molK'],selectors=['Pearson'],models=['Ridge'],raw=True,unit='at.%',group=None,k=6,budget=2,folds=3,test=.2,seed=42,optimizer='None')
r=client.post('/api/upload',json={'name':'demo.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()}); assert r.status_code==200,r.text; sid=r.json()['sid']; sig=r.json()['signature']
body=dict(rationale='Test',config={'targets':['YS','EL']},design=dict(directions={'YS':'max','EL':'max'},bounds={e:[5,45] for e in els},thresholds={'EL':20},pool_size=120,iterations=12))
for provider,url in [('openai','https://api.openai.com/v1'),('anthropic','https://api.anthropic.com/v1'),('deepseek','https://api.deepseek.com'),('qwen','https://dashscope.aliyuncs.com/compatible-mode/v1'),('gemini','https://generativelanguage.googleapis.com/v1beta/openai'),('ollama','http://127.0.0.1:11434/v1'),('custom','https://example.com/v1')]:
    mock=Mock(status_code=200); mock.json.return_value={'content':[{'type':'text','text':json.dumps(body)}]} if provider=='anthropic' else {'choices':[{'message':{'content':json.dumps(body)}}]}
    with patch('llm.requests.post',return_value=mock) as post:
        r=client.post('/api/plan/'+sid,json=dict(provider=provider,base_url=url,model='mock-model',api_key='test-key',config=c,goal='maximize strength and elongation',language='en')); assert r.status_code==200,r.text; p=r.json(); assert p['design']['thresholds']=={'EL':20}; assert 'test-key' not in r.text; assert post.call_args.args[0].endswith('/messages' if provider=='anthropic' else '/chat/completions'); print(provider,'mock PASS',flush=True)
# A newly generated plan deliberately invalidates the old adaptive plan ID.
r=client.post('/api/local-plan/'+sid,json={'config':c,'language':'en'}); assert r.status_code==200,r.text
assert client.post('/api/workflow/'+sid,json=dict(config=p['config'],design=p['design'],signature=sig,plan_id=p['plan_id'],approved=True)).status_code==400
with patch('app.request_plan',return_value=body):
    r=client.post('/api/plan/'+sid,json=dict(config=c,model='mock-model',goal='maximize strength and elongation',language='en')); assert r.status_code==200,r.text; p=r.json()
from regression_helpers import finish_stage2
p=finish_stage2(client,sid,p)
b=dict(config=p['config'],design=p['design'],signature=sig,plan_id=p['plan_id']); assert client.post('/api/workflow/'+sid,json=b).status_code==400; assert client.post('/api/workflow/'+sid,json=dict(b,approved=True,signature='wrong')).status_code==400
r=client.post('/api/workflow/'+sid,json=dict(b,approved=True)); assert r.status_code==200,r.text
for i in range(600):
    d=client.get('/api/status/'+sid).json()
    if d['status']!='running': break
    time.sleep(.15)
assert d['status']=='done',d; assert len(d['summary']['metrics'])==2; assert d['inverse']['evaluated']==d['inverse']['pool'] and d['inverse']['bo_evaluated']==12; assert d['inverse']['feasible']>=1; rows=pd.DataFrame(d['inverse']['rows']); assert (rows.loc[rows.Pareto,'pred_EL']>=20).all(); assert np.array_equal(rows.loc[rows.Feasible,'Pareto'].to_numpy(),pareto(rows.loc[rows.Feasible,['pred_YS','pred_EL']].to_numpy())); assert all(abs(rows[els].sum(axis=1)-100)<1e-7)
z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+sid).content)); provenance=json.loads(z.read('provenance.json')); assert provenance['approved_workflow']['signature']==sig; assert 'test-key' not in json.dumps(provenance); print('Automatic workflow PASS:',d['inverse']['pareto'],'Pareto solutions; export',len(z.namelist()),'entries',flush=True)
