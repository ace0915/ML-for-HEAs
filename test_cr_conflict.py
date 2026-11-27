import base64
from unittest.mock import patch
import numpy as np,pandas as pd
from fastapi.testclient import TestClient
from app import app,sessions
from mapping import identify
from engine import ELEMENTS,features
client=TestClient(app)
rng=np.random.default_rng(21)
x=rng.dirichlet([4]*5,36)*100
raw=pd.DataFrame(x,columns=['Fe(at%)','Ni(at%)','Co(at%)','Cr(at%)','Al(at%)'])
raw['CR(%)']=np.tile([0,20,40],12); raw['YS']=300+5*x[:,3]+4*raw['CR(%)'];raw['EL']=35-.2*x[:,4]-.1*raw['CR(%)']
def upload(d): return client.post('/api/upload',json={'name':'synthetic.csv','data':base64.b64encode(d.to_csv(index=False).encode()).decode()})
r=upload(raw);assert r.status_code==200,r.text;u=r.json();assert u['elements']==['Fe','Ni','Co','Cr','Al'];assert u['header_mapping']['Cr(at%)']=='Cr';assert 'CR(%)' not in u['header_mapping']; assert u['column_profile']['CR(%)']['role']=='process';assert u['column_profile']['CR(%)']['type']=='numeric';assert 'verify' in u['column_profile']['CR(%)']['warning'];assert sessions[u['sid']]['df']['CR(%)'].tolist()==raw['CR(%)'].tolist(); print('Exact reported conflict: upload preserves composition and processing PASS')
c=dict(elements=[],unit='auto',targets=['YS','EL'],extras=['CR(%)'],categorical_extras=[],descriptors=['S_mix_J_molK'],selectors=['Pearson'],models=['Ridge'],raw=True,group=None,k=5,budget=2,folds=3,test=.2,seed=42,optimizer='None')
with patch('app.request_plan',return_value={'rationale':'Mocked plan','config':{},'design':None}):
    r=client.post('/api/plan/'+u['sid'],json={'config':c,'model':'mock','language':'zh'});assert r.status_code==200,r.text
p=r.json();assert p['config']['extras']==['CR(%)'];assert any('CR(%)' in w for w in p['mapping_warnings']);assert p['config']['elements']==u['elements'];X=features(sessions[u['sid']]['df'],p['config']['elements'],'at.%',c['descriptors'],c['extras'],True,[]);assert np.allclose(X['CR(%)'],raw['CR(%)']);print('Mocked LLM planning + independent process feature PASS')
assert identify('Cr(%)',ELEMENTS)[0]=='Cr';assert identify('CR(at%)',ELEMENTS)==('Cr','at.%');assert identify('CR(wt%)',ELEMENTS)==('Cr','wt.%');assert identify('CR(%)',ELEMENTS)==(None,None);print('Explicit chromium labels remain supported PASS')
r=upload(raw.assign(Cr=raw['Cr(at%)']));assert r.status_code==400,r.text;print('Genuine duplicate chromium still rejected PASS')
r=upload(raw.drop(columns=['Cr(at%)']));assert r.status_code==200,r.text;assert r.json()['column_profile']['CR(%)']['role']=='review';assert 'Cr' not in r.json()['elements'];print('Standalone ambiguous CR requires review PASS')
r=upload(raw.assign(**{'AL(%)':raw['CR(%)']}));assert r.status_code==200,r.text;assert r.json()['column_profile']['AL(%)']['role']=='review';assert 'AL(%)' in r.json()['columns'];print('Other conflicting uppercase abbreviations preserved for review PASS')
