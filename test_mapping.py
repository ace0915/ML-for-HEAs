import base64,json
from unittest.mock import patch
import pandas as pd,numpy as np
from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
rng=np.random.default_rng(42)
raw=pd.DataFrame(rng.dirichlet([4,4,4],36)*100,columns=['Fe(at.%)','镍含量','Co_at.%']); raw['Processing']=['As-cast','Annealed']*18; raw['YS']=400+raw.iloc[:,0]*5; raw['EL']=30-raw.iloc[:,0]*.2

def upload(df):return client.post('/api/upload',json={'name':'test.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()})
def plan(u,c):
    with patch('app.request_plan',return_value={'rationale':'Mocked plan','config':{},'design':None}) as mock:
        r=client.post('/api/plan/'+u['sid'],json={'config':c,'model':'mock','language':'zh'});return r,mock
c=dict(elements=[],unit='auto',targets=['YS','EL'],extras=['Processing'],categorical_extras=['Processing'],descriptors=['S_mix_J_molK'],selectors=['Pearson'],models=['Ridge'],raw=True,group=None,k=5,budget=2,folds=3,test=.2,seed=42,optimizer='None')
u=upload(raw).json();assert u['elements']==['Fe','Ni','Co'],u;assert u['unit_suggestion']=='at.%';r,m=plan(u,c);assert r.status_code==200,r.text;assert r.json()['config']['elements']==['Fe','Ni','Co'];assert r.json()['config']['unit']=='at.%';assert m.called; print('Header aliases + empty selection fallback + planning PASS')
raw2=raw.rename(columns={'Fe(at.%)':'Fe','镍含量':'Ni','Co_at.%':'Co'});u=upload(raw2).json();assert u['unit_provisional'];r,m=plan(u,c);assert r.status_code==200,r.text;assert any('待审核单位' in w for w in r.json()['mapping_warnings']);print('Unlabelled percentage: explicit provisional warning PASS')
raw2[['Fe','Ni','Co']]/=100;u=upload(raw2).json();r,m=plan(u,c);assert r.status_code==200,r.text;assert r.json()['config']['unit']=='fraction';print('Fraction scale recognition PASS')
r=upload(raw.rename(columns={'Co_at.%':'Co(wt.%)'}));assert r.status_code==400,r.text;print('Mixed-unit rejection PASS')
r=upload(raw.assign(Fe=raw.iloc[:,0]));assert r.status_code==400,r.text;print('Duplicate element rejection PASS')
u=upload(pd.DataFrame({'foo':[1]*20,'YS':range(20)})).json();r,m=plan(u,dict(c,extras=[],categorical_extras=[]));assert r.status_code==400 and not m.called and '未识别到有效元素列' in r.text; print('Actionable missing element error PASS')
print('ALL MAPPING TESTS PASSED')
