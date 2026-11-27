"""Mixed composition/process tests using synthetic measurements, not material evidence."""
import base64, io, json, time, zipfile
from unittest.mock import Mock,patch
from fastapi.testclient import TestClient
from app import app, sessions
from engine import np,pd,features,train,pareto
from process import column_profile
import joblib
client=TestClient(app); rng=np.random.default_rng(43); els=['Fe','Ni','Co','Cr','Al']; n=96
x=rng.dirichlet([4]*5,n)*100; df=pd.DataFrame(x,columns=els)
conditions=[('As-cast',25,0,1),('Annealed',800,0,2),('Cold-rolled',25,40,3)]
fields=['Processing','AnnealingTemperature','RollingReduction','ProcessingCode']; df[fields]=[conditions[i%3] for i in range(n)]
df['YS']=250+10*x[:,4]+df.RollingReduction*5+(df.ProcessingCode-1)*40; df['EL']=45-.4*x[:,4]-df.RollingReduction*.2
# Explicit missingness in process fields, not synthetic zero or arbitrary process code.
df.loc[0,'Processing']=None; df.loc[1,'AnnealingTemperature']=np.nan; df['Sample_ID']=[f'S{i}' for i in range(n)]
r=client.post('/api/upload',json={'name':'mixed_synthetic.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()}); assert r.status_code==200,r.text
u=r.json(); sid=u['sid']; prof=u['column_profile']; assert all(prof[e]['role']=='process' for e in fields); assert prof['Processing']['type']==prof['ProcessingCode']['type']=='categorical'; assert prof['AnnealingTemperature']['type']==prof['RollingReduction']['type']=='numeric'; assert prof['Sample_ID']['role']=='metadata'; assert prof['YS']['role']=='target'; print('Role & type recognition PASS')
c=dict(elements=els,targets=['YS','EL'],extras=fields,categorical_extras=['Processing','ProcessingCode'],descriptors=['S_mix_J_molK'],selectors=['Pearson'],models=['Ridge'],raw=True,unit='at.%',group=None,k=20,budget=2,folds=3,test=.2,seed=42,optimizer='None')
r=client.post('/api/analyze/'+sid,json=c); assert r.status_code==200,r.text; assert r.json()['missing']['Processing']==1; assert r.json()['missing']['AnnealingTemperature']==1; print('Mixed feature analysis PASS')
wrong=dict(c,categorical_extras=[]); r=client.post('/api/train/'+sid,json=wrong); assert r.status_code==400,r.text
r=client.post('/api/local-plan/'+sid,json={'config':c}); assert r.status_code==200,r.text; p=r.json(); assert p['design']['conditions']['Processing'] in [a[0] for a in conditions]; assert isinstance(p['design']['conditions']['ProcessingCode'],str)
body={'rationale':'Mixed inputs','config':{'extras':fields,'categorical_extras':c['categorical_extras']},'design':{'process_mode':'observed','directions':{'YS':'max','EL':'max'},'pool_size':80,'iterations':12,'thresholds':{'EL':15}}}
mock=Mock(status_code=200); mock.json.return_value={'choices':[{'message':{'content':json.dumps(body)}}]}
with patch('llm.requests.post',return_value=mock) as post:
    r=client.post('/api/plan/'+sid,json={'config':c,'model':'mock-model','provider':'openai','api_key':'test-key','goal':'Joint strength and elongation and process optimization'}); assert r.status_code==200,r.text; p=r.json(); payload=post.call_args.kwargs['json']; assert 'column_profile' in payload['messages'][1]['content']; assert p['config']['categorical_extras']==c['categorical_extras']; assert p['design']['conditions']=={}
print('Process-aware local and mocked LLM planning PASS')
def wait():
    for _ in range(600):
        st=client.get('/api/status/'+sid).json()
        if st['status']!='running': assert st['status']=='done',st; return st
        time.sleep(.15)
    raise RuntimeError('timeout')
from regression_helpers import finish_stage2
p=finish_stage2(client,sid,p)
r=client.post('/api/workflow/'+sid,json={'approved':True,'signature':u['signature'],'plan_id':p['plan_id'],'config':p['config'],'design':p['design']}); assert r.status_code==200,r.text
st=wait(); assert len(st['summary']['metrics'])==2; assert st['summary']['process_profile']['observed_combinations']==3; assert st['inverse']['evaluated']==st['inverse']['pool'] and st['inverse']['bo_evaluated']==12; assert st['inverse']['pareto']>0
out=pd.DataFrame(st['inverse']['rows']); tuples=set(map(tuple,out[fields].assign(ProcessingCode=lambda x:x.ProcessingCode.astype(int)).values)); assert tuples<=set(conditions); assert len(tuples)>1; assert all(abs(out[els].sum(axis=1)-100)<1e-8); print('Human-approved mixed workflow + observed joint process search PASS')
for t,res in sessions[sid]['results'].items():
    assert any(x.startswith('cat__Processing_') for x in res['encoded_cols']); assert any(x.startswith('cat__ProcessingCode_') for x in res['encoded_cols']); assert 'num__AnnealingTemperature' in res['encoded_cols']; assert set(res['train_rows']).isdisjoint(res['test_rows']);
    X=features(df,els,'at.%',p['config']['descriptors'],fields,p['config']['raw'],p['config']['categorical_extras']); unknown=X.iloc[:1].copy(); unknown['Processing']='Unseen-Route'; assert np.isfinite(res['model'].predict(unknown[res['cols']])).all()
print('Train-fold categorical encoding and unknown category inference PASS')
d=dict(p['design'],process_mode='fixed',conditions={'Processing':'Annealed','AnnealingTemperature':800,'RollingReduction':0,'ProcessingCode':'2'},iterations=8)
r=client.post('/api/inverse/'+sid,json=dict(d,conditions=dict(d['conditions'],Processing='Invented'))); assert r.status_code==400,r.text
r=client.post('/api/inverse/'+sid,json=dict(d,conditions=dict(d['conditions'],AnnealingTemperature=9999))); assert r.status_code==400,r.text
r=client.post('/api/inverse/'+sid,json=d); assert r.status_code==200,r.text; st=wait(); assert all(z['Processing']=='Annealed' and z['AnnealingTemperature']==800 and z['ProcessingCode']=='2' for z in st['inverse']['rows']); print('Fixed mixed processing conditions and constraint rejection PASS')
z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+sid).content)); assert 'figures/process_1_distribution.png' in z.namelist(); split=json.loads(z.read('models/1_split.json')); assert any(n.startswith('cat__') for n in split['encoded_features']); model=joblib.load(io.BytesIO(z.read('models/1.joblib'))); assert np.isfinite(model.predict(X.iloc[:2][split['features']])).all(); print('Export and serialized pipeline round-trip PASS')
print('ALL PROCESS TESTS PASSED')
