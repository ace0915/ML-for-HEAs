"""Adapt legacy integration cases to the V4.9 second approval prerequisite."""
from unittest.mock import patch
from stage_two import TRAIN_KEYS

def finish_stage2(client,sid,plan,config=None,design=None):
    c=config or plan['config']; d=plan['design'] if design is None else design
    reply={'rationale':'Regression mock stage 2 retains explicit test settings','config':{'target_settings':{t:{k:v[k] for k in TRAIN_KEYS} for t,v in c['target_settings'].items()}},'design':None}
    with patch('app.request_plan',return_value=reply):
        r=client.post('/api/plan-stage2/'+sid,json=dict(config=c,design=d,plan_id=plan['plan_id'],model=plan['model'],provider=plan['provider'],language='en')); assert r.status_code==200,r.text
    return r.json()
