"""Exercise real Chromium UI with actual FastAPI routes, using intercepted HTTP transport."""
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from app import app
from unittest.mock import patch
import json, numpy as np, pandas as pd, base64, time
from pathlib import Path
client=TestClient(app)
errors=[]; requests=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1550,'height':1100})
    def bridge(path,body=None):
        if body: requests.append((path,json.loads(body)))
        res=client.request('POST' if body else 'GET',path,content=body,headers={'Content-Type':'application/json'} if body else {})
        return {'status':res.status_code,'data':res.json()}
    page.expose_function('testApiBridge',bridge)
    page.add_init_script("""window.fetch = async (path,opts={}) => {let r=await window.testApiBridge(path,opts.body||null);return {ok:r.status>=200&&r.status<300,json:async()=>r.data}};""")
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('dialog',lambda d:d.accept())
    page.set_content(Path('index.html').read_text().replace('<script>','<script>Object.defineProperty(window,"localStorage",{value:{getItem:()=>null,setItem:()=>{}}}); window.fetch=async (path,opts={})=>{let r=await window.testApiBridge(path,opts.body||null);return {ok:r.status>=200&&r.status<300,json:async()=>r.data}};\n',1))
    page.wait_for_function("document.getElementById('serverVersion').textContent.includes('4.9.0') && document.querySelectorAll('#modelOptions input').length===30")
    rng=np.random.default_rng(42)
    def dataset(n):
        x=rng.dirichlet([4]*5,n)*100;df=pd.DataFrame(x,columns=['Fe','Ni','Co','Cr','Al']);df['CR(%)']=np.arange(n)%3*20;df['YS']=250+8*x[:,4]+4*df['CR(%)'];df['EL']=30-.2*x[:,3];return {'name':'test.csv','data':base64.b64encode(df.to_csv(index=False).encode()).decode()}
    def plan_for(n):
        page.evaluate('(b)=>loadData(b)',dataset(n));page.evaluate("document.getElementById('unit').value='at.%'; go('agent')")
        page.locator('#adaptiveBtn').click();page.wait_for_function("document.querySelectorAll('#adaptiveReport .adaptive-card').length>=2",timeout=60000)
        return json.loads(page.locator('#approvedPlan').input_value())
    a=plan_for(48);b=plan_for(600)
    assert a['config']['target_settings']['YS']['models']!=b['config']['target_settings']['YS']['models']
    assert a['config']['target_settings']['YS']['budget']!=b['config']['target_settings']['YS']['budget']
    page.screenshot(path='/mnt/data/EntropyForge_V4_9_Adaptive_CN.png',full_page=True)
    page.locator('#langEn').click();assert 'Target-specific adaptive execution plan' in page.locator('#adaptiveReport').inner_text()
    page.screenshot(path='/mnt/data/EntropyForge_V4_9_Adaptive_EN.png',full_page=True)
    page.evaluate("go('models')");page.wait_for_function("document.querySelectorAll('#adaptiveTraining .adaptive-card').length>=2")
    assert 'Shortlist' in page.locator('#adaptiveTraining').inner_text()
    # LLM explicit target settings, using mock provider response and real backend merge.
    page.evaluate("go('agent');document.getElementById('llm_model').value='mock'")
    settings={t:dict(models=['Ridge','Extra trees'] if t=='YS' else ['Ridge','Huber'],selectors=['Pearson'],k=4,optimizer='None',budget=1,folds=2,screen_top=2) for t in b['config']['targets']}
    with patch('app.request_plan',return_value={'rationale':'Mock explicit settings','config':{'target_settings':settings,'descriptors':[]},'design':None}):
        page.locator('#planBtn').click();page.wait_for_function("document.getElementById('planout').textContent.includes('Mock explicit settings')")
    actual=json.loads(page.locator('#approvedPlan').input_value());assert actual['config']['target_settings']==settings
    from stage_two import TRAIN_KEYS
    with patch('app.request_plan',return_value={'rationale':'Mock second stage','config':{'target_settings':{t:{k:v[k] for k in TRAIN_KEYS} for t,v in settings.items()}},'design':None}):
        page.locator('#stage2Btn').click();page.wait_for_function("document.getElementById('stage2Report').textContent.includes('Mock second stage')")
    page.locator('#approveCheck').check();page.locator('button[onclick="applyAndRun()"]').click()
    page.wait_for_function("document.getElementById('status').textContent.includes('完成') || document.getElementById('status').textContent.includes('completed') || document.getElementById('status').textContent.includes('failed')",timeout=90000)
    sent=[b for path,b in requests if path.startswith('/api/workflow/')][-1];assert sent['config']['target_settings']==settings
    sid=page.evaluate('sid');res=client.get('/api/status/'+sid).json();assert res['status']=='done',res;assert res['summary']['target_settings']==settings
    page.evaluate("go('design')")
    page.wait_for_selector('#processMode',state='attached')
    page.locator('#processMode').select_option('bounded'); assert page.locator('#rangeProcessFields').is_visible()
    index=actual['config']['extras'].index('CR(%)')
    page.locator('#rangeLo'+str(index)).fill('10');page.locator('#rangeHi'+str(index)).fill('30')
    page.locator('#pool_size').fill('30');page.locator('#iterations').fill('8')
    page.locator('button[onclick="runDesign()"]').click()
    page.wait_for_function("document.getElementById('designout').textContent.includes('Pareto')",timeout=90000)
    payload=[b for path,b in requests if path.startswith('/api/inverse/')][-1];assert payload['process_ranges']=={'CR(%)':[10,30]} and payload['process_mode']=='bounded'
    st=client.get('/api/status/'+sid).json();assert st['status']=='done' and all(10<=r['CR(%)']<=30 for r in st['inverse']['rows'])
    assert not errors,errors
    browser.close()
print('PASS: actual Chromium UI + real FastAPI routes (injected local API bridge): version, 48/600 data differences, no-API adaptive plan, bilingual rendering, training page display, mocked LLM per-target plan, approval request and actual training configuration match, bounded process form -> validation -> inverse request -> candidates within range.')
