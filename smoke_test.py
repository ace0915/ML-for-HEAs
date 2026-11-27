"""Local integration test: synthetic data, no real material performance claims."""
from fastapi.testclient import TestClient
from app import app
from engine import *
import base64,time,zipfile,io
client=TestClient(app); rng=np.random.default_rng(42)
x=rng.dirichlet([3,3,3,3,3],size=90)*100
els=['Fe','Ni','Co','Cr','Al']; df=pd.DataFrame(x,columns=els); df['YS']=250+12*x[:,4]+5*x[:,3]+rng.normal(0,8,90); df['EL']=45-.7*x[:,4]+rng.normal(0,1,90)
assert client.get('/').status_code==200
reg=client.get('/api/options').json(); assert [len(reg[k]) for k in ['descriptors','selectors','models']]==[30,10,30]
r=client.post('/api/upload',json={'name':'synthetic.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()}); assert r.status_code==200,r.text
sid=r.json()['sid']; c=dict(elements=els,targets=['YS','EL'],extras=[],descriptors=DESCRIPTORS[:4],selectors=['Pearson','Extra trees importance'],models=['Ridge','Extra trees'],raw=True,unit='at.%',group=None,k=6,budget=3,folds=3,test=.2,seed=42,optimizer='Genetic algorithm')
r=client.post('/api/analyze/'+sid,json=c); assert r.status_code==200,r.text
r=client.post('/api/train/'+sid,json=c); assert r.status_code==200,r.text
def wait():
    for _ in range(240):
        d=client.get('/api/status/'+sid).json()
        if d['status']!='running':
            assert d['status']=='done',d; return d
        time.sleep(.25)
    raise RuntimeError('timeout')
d=wait(); assert len(d['summary']['metrics'])==2; print('Training:',d['summary']['metrics'])
r=client.post('/api/inverse/'+sid,json=dict(bounds={e:[5,45] for e in els},conditions={},directions={'YS':'max','EL':'max'},step=1,pool_size=150,iterations=15,min_entropy=8)); assert r.status_code==200,r.text
d=wait(); assert d['inverse']['evaluated']==d['inverse']['pool'] and d['inverse']['bo_evaluated']==15 and d['inverse']['pareto']>=1; assert all(abs(sum(r[e] for e in els)-100)<1e-8 for r in d['inverse']['rows']); print('Inverse:',{k:v for k,v in d['inverse'].items() if k!='rows'})
r=client.get('/api/export/'+sid); assert r.status_code==200; z=zipfile.ZipFile(io.BytesIO(r.content)); assert 'provenance.json' in z.namelist() and 'design.csv' in z.namelist() and 'models/1.joblib' in z.namelist(); print('ZIP entries:',len(z.namelist()))
for m in OPTIMIZERS[1:]:
    p,t=search({'a':[0,1,2],'b':[0,1,2]},lambda p:(p['a']-1)**2+(p['b']-2)**2,m,9,42); assert p=={'a':1,'b':2},(m,p); print(m,'PASS')
print('PASS')
