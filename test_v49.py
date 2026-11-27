import base64,copy,io,time,zipfile
from unittest.mock import patch
import numpy as np,pandas as pd
from fastapi.testclient import TestClient
from app import app,sessions,get_features
from engine import split_target
from stage_two import selected_profiles
from single_objective import single_results
rng=np.random.default_rng(41); a=rng.dirichlet([5]*4,60)*100; df=pd.DataFrame(a,columns=['Al','Fe','Co','Ni']);df['Temp']=rng.integers(700,901,60);df['HV']=300+3*a[:,0]+.2*df.Temp;df['EL']=40-.3*a[:,1]
c=dict(elements=['Al','Fe','Co','Ni'],targets=['HV','EL'],extras=['Temp'],categorical_extras=[],descriptors=[],selectors=['Pearson'],models=['Ridge'],raw=True,unit='at.%',group=None,k=3,budget=1,folds=2,test=.2,seed=42,optimizer='None')
client=TestClient(app)
def post(url,body):
 r=client.post(url,json=body);assert r.status_code==200,r.text;return r.json()
u=post('/api/upload',dict(name='synthetic.csv',data=base64.b64encode(df.to_csv(index=False).encode()).decode()));sid=u['sid']
settings={t:dict(models=['Ridge'],selectors=['Pearson'],k=k,optimizer='None',budget=1,folds=2,screen_top=1) for t,k in [('HV',2),('EL',3)]}
first=dict(rationale='stage 1',config={'target_settings':settings,'descriptors':[]},design={'directions':{'HV':'max'},'pool_size':40,'iterations':3,'top_n':7,'plot_axes':{'x_element':'Al','y_target':'HV'}})
with patch('app.request_plan',return_value=first): p=post('/api/plan/'+sid,dict(config=c,model='mock',language='en',goal='maximize HV; plot Al'))
assert p['planning_stage']==1
body=dict(config=p['config'],design=p['design'],approved=True,signature=u['signature'],plan_id=p['plan_id']);assert client.post('/api/workflow/'+sid,json=body).status_code==400
def stage2(b,profile,registry):
 assert b['stage']==2
 q=profile['selected_training_profiles'];assert q['HV']['effective_inputs']==2 and q['EL']['effective_inputs']==3
 assert len(q['HV']['selected_features'])==2 and 'mean_abs_interfeature_corr' in q['HV']
 return dict(rationale='stage 2 based on selected features',config={'target_settings':{'HV':{'models':['Extra trees'],'optimizer':'None','budget':1,'folds':2,'screen_top':1},'EL':{'models':['Ridge'],'optimizer':'None','budget':1,'folds':2,'screen_top':1}}},design=None)
with patch('app.request_plan',side_effect=stage2) as mock: q=post('/api/plan-stage2/'+sid,dict(config=p['config'],design=p['design'],plan_id=p['plan_id'],model='mock',language='en'));assert mock.call_count==1
assert q['planning_stage']==2 and q['design']==p['design'];assert q['config']['target_settings']['HV']['models']==['Extra trees']
frame=sessions[sid]['df']; profiles,_=selected_profiles(frame,get_features(sessions[sid],q['config']),q['config']); changed=frame.copy()
for t in c['targets']:
 ix,tr,te,_=split_target(frame,t,None,q['config']['test'],42,2);changed.loc[ix[te],t]+=1e8
profiles2,_=selected_profiles(changed,get_features(sessions[sid],q['config']),q['config']);assert profiles==profiles2
body.update(config=q['config'],design=q['design'],plan_id=q['plan_id']);bad=copy.deepcopy(body);bad['config']['target_settings']['HV']['k']=1;assert client.post('/api/workflow/'+sid,json=bad).status_code==400
post('/api/workflow/'+sid,body)
for i in range(600):
 r=client.get('/api/status/'+sid).json()
 if r['status']!='running':break
 time.sleep(.1)
assert r['status']=='done',r
for t,k in [('HV',2),('EL',3)]:
 m=sessions[sid]['results'][t]['model'];assert m.named_steps['model'].n_features_in_==k
inv=r['inverse'];assert inv['mode']=='single' and inv['evaluated']==inv['pool'];assert len(inv['top_recommendations'])==7 and inv['envelope'];assert inv['top_recommendations'][0]['pred_HV']==max(x['pred_HV'] for x in inv['rows'] if x['Feasible']);assert 'envelope_single_objective' in inv['figures']
z=zipfile.ZipFile(io.BytesIO(client.get('/api/export/'+sid).content));assert all(x in z.namelist() for x in ['single_ranking.csv','top_recommendations.csv','optimal_envelope.csv','figures/envelope_single_objective.png']);assert b'selected_training_profiles' in z.read('provenance.json');print('Stage 1/2, selected-feature profile, holdout isolation, stale K rejection, actual trained inputs, single-objective envelope, ZIP PASS',flush=True)
f=pd.DataFrame({'Temp':[700,700,750,800,800],'pred_HV':[4,1,2,3,0],'Feasible':[True,True,True,False,True],'pool_index':range(5)})
r,t,e=single_results(f,'HV','min',2,{'x_process':'Temp','y_target':'HV','bin_width':50});assert t.pred_HV.tolist()==[0,1] and e.pred_HV.tolist()==[1,2,0];assert len(single_results(f.assign(Feasible=False),'HV','min',2)[1])==0
print('Minimization, process grouping and empty feasible pool PASS',flush=True)
