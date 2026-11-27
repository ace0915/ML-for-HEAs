from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import base64, io, json, hashlib, time, uuid, threading, traceback, os, zipfile, re
from pathlib import Path
import numpy as np, pandas as pd, requests, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from engine import *
from process import column_profile, categorical_columns, process_summary, constrained_process_pool
from mapping import normalize_headers, proposed_unit
from adaptive import baseline, merge_adaptive, validate_target_settings
VERSION='4.9.0'
from stage_two import feature_signature, selected_profiles, training_baseline, merge_stage_two
from single_objective import single_results
import copy
app=FastAPI(title='EntropyForge Web',version=VERSION)
ROOT=Path(__file__).parent
sessions={}; lock=threading.Lock()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':15,'font.weight':'bold','axes.labelsize':17,'axes.labelweight':'bold','axes.linewidth':1.4,'axes.grid':False,'xtick.direction':'in','ytick.direction':'in','xtick.major.width':1.4,'ytick.major.width':1.4,'xtick.labelsize':14,'ytick.labelsize':14,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':120,'savefig.dpi':300,'legend.frameon':False})
COLORS=['#3A6F9D','#D17A58','#43958B','#8E7EAD','#D1A34B']
def jsonable(a):
    if isinstance(a,pd.DataFrame): return json.loads(a.to_json(orient='records',date_format='iso'))
    if isinstance(a,np.ndarray): return jsonable(a.tolist())
    if isinstance(a,np.generic): return jsonable(a.item())
    if isinstance(a,float) and not np.isfinite(a): return None
    if isinstance(a,dict): return {str(k):jsonable(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)): return [jsonable(v) for v in a]
    return a

def session(sid):
    if sid not in sessions: raise HTTPException(404,'请先上传数据。')
    return sessions[sid]
def log(s,msg): s['logs'].append({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'message':msg})
def check(s,c):
    cols=set(s['df'].columns); els=c.get('elements',[]); targets=c.get('targets',[]); extras=c.get('extras',[]); desc=c.get('descriptors',[])
    if not els or not set(els)<=set(ELEMENTS) or not set(els)<=cols: raise ValueError('请选择以标准元素符号命名的成分列。')
    if not targets or not set(targets)<=cols or not set(extras)<=cols or len(set(els+targets+extras))!=len(els+targets+extras): raise ValueError('目标、元素与工艺特征必须存在且互不重叠。')
    if not set(desc)<=set(DESCRIPTORS) or not set(c.get('selectors',[]))<=set(SELECTORS) or not set(c.get('models',[]))<=set(model_specs()): raise ValueError('配置包含未注册模块。')
    if c.get('unit') not in ['at.%','wt.%','fraction']: raise ValueError('请选择并确认成分单位：at.%、wt.% 或 fraction。 / Select composition units.')
    hints={s.get('unit_hints',{}).get(e) for e in els}-{None}
    if hints and hints!={c['unit']}: raise ValueError('Selected unit conflicts with composition header units / 所选单位与表头标注不一致。')
    if c.get('group') and (c['group'] not in cols or c['group'] in els+targets+extras): raise ValueError('分组列不能是输入或目标列。')
    if set(desc)&set(extras): raise ValueError('描述符名称与额外输入列重复。')
    if not(c.get('raw') or desc or extras): raise ValueError('至少选一种输入特征。')
    if c.get('optimizer','None') not in OPTIMIZERS: raise ValueError('优化器不支持。')
    categorical_columns(s['df'],c)
    return els,targets,extras,desc

def get_features(s,c):
    els,targets,extras,desc=check(s,c)
    return features(s['df'],els,c['unit'],desc,extras,c.get('raw',True),categorical_columns(s['df'],c))

@app.get('/',response_class=HTMLResponse)
def home(): return HTMLResponse((ROOT/'index.html').read_text(encoding='utf-8'),headers={'Cache-Control':'no-store, no-cache, must-revalidate','X-EntropyForge-Version':VERSION})
@app.get('/api/version')
def version(): return {'version':VERSION,'adaptive_preview':True,'target_specific_execution':True,'two_stage_planning':True,'single_objective_envelope':True}
@app.get('/api/options')
def options(): return dict(descriptors=DESCRIPTORS,selectors=SELECTORS,models=list(model_specs()),optimizers=OPTIMIZERS)
@app.post('/api/upload')
def upload(body:dict):
    try:
        b=base64.b64decode(body['data']); name=body['name']; ext=Path(name).suffix.lower()
        if len(b)>30*1024**2: raise ValueError('文件最大30MB。')
        if ext=='.csv':
            try: df=pd.read_csv(io.BytesIO(b),encoding='utf-8-sig')
            except UnicodeDecodeError: df=pd.read_csv(io.BytesIO(b),encoding='gb18030')
        elif ext=='.xlsx': df=pd.read_excel(io.BytesIO(b),sheet_name=body.get('sheet') or 0)
        else: raise ValueError('支持 CSV / XLSX。')
        df.columns=[str(x).strip() for x in df.columns]; df=df.dropna(how='all').reset_index(drop=True)
        if df.columns.duplicated().any() or len(df)==0: raise ValueError('列名重复或数据为空。')
        original_df=df.copy(); df,header_mapping,unit_hints=normalize_headers(df,ELEMENTS); detected=[x for x in df if x in ELEMENTS]; unit_suggestion,unit_provisional=proposed_unit(df,detected,unit_hints); sid=uuid.uuid4().hex; s=dict(df=df,original_df=original_df,header_mapping=header_mapping,unit_hints=unit_hints,signature=hashlib.sha256(b).hexdigest(),filename=name,logs=[],status='ready',plots={},plot_files={}); sessions[sid]=s; log(s,f'上传 {name} / SHA256 {s["signature"]}')
        return dict(sid=sid,rows=len(df),columns=list(df.columns),elements=[x for x in df if x in ELEMENTS],preview=jsonable(df.head(10)),missing={c:int(df[c].isna().sum()) for c in df},signature=s['signature'],column_profile=column_profile(df,detected),header_mapping=header_mapping,unit_suggestion=unit_suggestion,unit_provisional=unit_provisional)
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/status/{sid}')
def status(sid:str):
    s=session(sid); return jsonable({'status':s['status'],'logs':s['logs'][-100:],'error':s.get('error'),'summary':s.get('summary'),'inverse':s.get('inverse_summary')})

def save_plot(s,key,fig):
    for ext in ('png','svg','pdf'):
        b=io.BytesIO(); fig.savefig(b,format=ext,bbox_inches='tight'); s['plot_files'][key+'.'+ext]=b.getvalue()
    s['plots'][key]=base64.b64encode(s['plot_files'][key+'.png']).decode(); plt.close(fig)
def train_plots(s):
    for i,(t,r) in enumerate(s['results'].items()):
        d=r['holdout']; a=d.actual; p=d.predicted; lo=min(a.min(),p.min()); hi=max(a.max(),p.max()); pad=max((hi-lo)*.06,.01)
        fig,ax=plt.subplots(figsize=(5.6,5.3)); ax.scatter(a,p,s=56,color=COLORS[0],edgecolors='white',linewidth=.6); ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#555555',lw=1.2); ax.set(xlabel=f'Experimental {t}',ylabel=f'Predicted {t}',xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad)); ax.text(.04,.96,f"$R^2$ = {r['metrics']['R2']:.4f}\nRMSE = {r['metrics']['RMSE']:.3f}",transform=ax.transAxes,va='top',fontsize=13); save_plot(s,f'{i+1}_parity',fig)
        fig,ax=plt.subplots(figsize=(5.8,4.4)); ax.scatter(p,a-p,s=45,color=COLORS[2],edgecolors='white',linewidth=.5); ax.axhline(0,ls='--',color='#555555'); ax.set(xlabel=f'Predicted {t}',ylabel='Residual'); save_plot(s,f'{i+1}_residual',fig)
        d=s['comparison']; d=d[d.target==t].copy(); d['model']=d['model']+d['stage'].map({'screen':' (screen)','final':' (final)'}).fillna(''); d=d.sort_values('cv_rmse',ascending=False); fig,ax=plt.subplots(figsize=(7,max(3.5,len(d)*.38))); ax.barh(d.model,d.cv_rmse,color=[COLORS[1] if x else COLORS[0] for x in d.selected]); ax.set_xlabel('Training CV RMSE'); save_plot(s,f'{i+1}_comparison',fig)
        d=r['ranking'].sort_values('score').tail(20); fig,ax=plt.subplots(figsize=(7,max(3.5,len(d)*.36))); ax.barh(d.display_feature,d.score,color=[COLORS[2] if x else '#B8C7D2' for x in d.selected]); ax.set_xlabel('Consensus rank score'); ax.set_xlim(0,1.05);
        if r['ranking'].score.nunique()==1: ax.set_title('Tied scores (computed)',fontsize=12);
        save_plot(s,f'{i+1}_features',fig)
        tr=s['traces'];
        if len(tr):
            tr=tr[(tr.target==t)&(tr.model==r['name'])];
            if len(tr):
                fig,ax=plt.subplots(figsize=(6,4)); ax.plot(tr.trial,tr.cv_rmse.replace(np.inf,np.nan).cummin(),'-o',color=COLORS[1]); ax.set(xlabel='Trial',ylabel='Best CV RMSE'); save_plot(s,f'{i+1}_optimization',fig)

def analysis_plots(s,X,c):
    corr=X.corr(numeric_only=True); n=min(len(corr),30); corr=corr.iloc[:n,:n]
    if n:
        fig,ax=plt.subplots(figsize=(max(6,n*.37),max(5,n*.37))); im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r'); ax.set_xticks(range(n),corr.columns,rotation=90,fontsize=8); ax.set_yticks(range(n),corr.index,fontsize=8); fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Pearson r'); save_plot(s,'data_correlation',fig)
    for i,e in enumerate(c['extras']):
        fig,ax=plt.subplots(figsize=(6,4.3))
        if e in c.get('categorical_extras',[]):
            counts=X[e].fillna('(missing)').value_counts().head(12).sort_values(); ax.barh(counts.index,counts.values,color=COLORS[2]); ax.set_xlabel('Count'); ax.set_ylabel(e)
        else:
            ax.hist(X[e].dropna(),bins='auto',color=COLORS[2],edgecolor='white'); ax.set(xlabel=e,ylabel='Count')
        save_plot(s,f'process_{i+1}_distribution',fig)
    for i,t in enumerate(c['targets']):
        y=pd.to_numeric(s['df'][t],errors='coerce').dropna(); fig,ax=plt.subplots(figsize=(5.8,4.3)); ax.hist(y,bins='auto',color=COLORS[0],edgecolor='white',linewidth=.8); ax.set(xlabel=t,ylabel='Count'); save_plot(s,f'data_{i+1}_distribution',fig)
    x=fractions(s['df'],c['elements'],c['unit'])*100
    fig,ax=plt.subplots(figsize=(max(6,len(c['elements'])*.6),4.5)); ax.boxplot(x,patch_artist=True,boxprops={'facecolor':'#B6CFDA'},medianprops={'color':'#9D563F','linewidth':1.5}); ax.set_xticks(range(1,len(c['elements'])+1),c['elements'],rotation=45 if len(c['elements'])>10 else 0); ax.set_ylabel('Composition (at.%)'); save_plot(s,'data_composition_coverage',fig)

def run_job(s,fn):
    try:
        fn(); s['status']='done'; log(s,'执行完成')
    except Exception as e:
        s['status']='error'; s['error']=str(e); log(s,'执行失败: '+str(e)); traceback.print_exc()
def start(s,fn):
    with lock:
        if s['status']=='running': raise HTTPException(409,'已有任务执行中。')
        s['status']='running'; s['error']=None; threading.Thread(target=run_job,args=(s,fn),daemon=True).start()
    return {'status':'running'}
@app.post('/api/analyze/{sid}')
def analyze(sid:str,c:dict):
    s=session(sid)
    try:
        if s['status']=='running': raise ValueError('任务执行中。')
        X=get_features(s,c); s['feature_table']=X; s['feature_config']=c; log(s,f'生成 {X.shape[1]} 个输入特征')
        return jsonable(dict(preview=X.head(10),summary=X.describe(include="all").T.reset_index(),missing={k:int(X[k].isna().sum()) for k in X},correlation=X.corr(numeric_only=True).round(3).reset_index()))
    except Exception as e: raise HTTPException(400,str(e))
def validate_train(s,c):
    check(s,c)
    validate_target_settings(c)
    if not c.get('selectors') or len(set(c['selectors']))!=len(c['selectors']): raise ValueError('Feature evaluation and Top-K selection are mandatory; select at least one distinct method. / 必须选择至少一种特征评价方法。')
    if not c.get('models'): raise ValueError('至少选择一个模型。')
    if type(c.get('k')) is not int: raise ValueError('Top-K must be an integer / 保留特征数必须为整数。')
    if not 2<=int(c['folds'])<=10 or not .1<=float(c['test'])<=.4 or not 1<=int(c['budget'])<=100 or not 1<=int(c['k'])<=100 or not 0<=int(c['seed'])<=2**32-1: raise ValueError('训练参数超出范围。')
    for t in c['targets']:
        if np.isfinite(pd.to_numeric(s['df'][t],errors='coerce')).sum()<12: raise ValueError(f'{t}: 有效样本不足12条。')
    fractions(s['df'],c['elements'],c['unit']); process_summary(s['df'],c)

def train_stage(s,c):
    log(s,'STAGE 1/3 · Descriptor engineering and feature selection')
    c=dict(c,categorical_extras=categorical_columns(s['df'],c)); X=get_features(s,c); ps=process_summary(s['df'],c); log(s,f'Process features: {len(c["extras"])} ({len(c["categorical_extras"])} categorical)'); s.update(plots={},plot_files={},config=c,feature_table=X,feature_config=c); s.pop('inverse_summary',None); s.pop('design',None); s.pop('history',None); s.pop('single_ranking',None); s.pop('top_recommendations',None); s.pop('optimal_envelope',None); s.pop('summary',None); s.pop('results',None)
    r,comp,fail,traces=train(s['df'],X,c['targets'],c['models'],c['selectors'],int(c['k']),c['optimizer'],int(c['budget']),int(c['folds']),float(c['test']),int(c['seed']),c.get('group'),lambda msg:log(s,msg),target_settings=c.get('target_settings')); s.update(results=r,comparison=comp,failures=fail,traces=traces)
    train_plots(s); analysis_plots(s,X,c); normalized=fractions(s['df'],c['elements'],c['unit'])*100; s['summary']=dict(default_bounds={e:[float(np.floor(normalized[:,i].min())),float(np.ceil(normalized[:,i].max()))] for i,e in enumerate(c['elements'])},default_conditions=ps['default'],process_profile=ps,metrics=[dict(target=t,model=v['name'],cv_rmse=v['cv_rmse'],n_train=len(v['train_rows']),n_test=len(v['test_rows']),**v['metrics']) for t,v in r.items()],comparison=jsonable(comp),failures=jsonable(fail),figures=list(s['plots']),target_settings=c.get('target_settings'))
    log(s,'STAGE 2/3 · Training and independent holdout evaluation completed')

@app.post('/api/feature-preview/{sid}')
def feature_preview(sid:str,c:dict):
    s=session(sid)
    try:
        if s['status']=='running': raise ValueError('A workflow is running')
        validate_train(s,c)
        return jsonable(preview_features(s['df'],get_features(s,c),c))
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/api/train/{sid}')
def training(sid:str,c:dict):
    s=session(sid)
    try: validate_train(s,c)
    except Exception as e: raise HTTPException(400,str(e))
    return start(s,lambda:train_stage(s,c))

def validate_design(c,b,df):
    dirs=b['directions']; bounds=b['bounds']; cond=b.get('conditions',{}); limits=b.get('thresholds',{})
    if not isinstance(dirs,dict) or not dirs or not set(dirs)<=set(c['targets']) or any(v not in ('max','min') for v in dirs.values()): raise ValueError('Invalid objective directions')
    mode=b.get('process_mode','fixed'); ps=process_summary(df,c)
    ranges=b.get('process_ranges',{}); categories=b.get('process_categories',{})
    for name,value in [('conditions',cond),('thresholds',limits),('process_ranges',ranges),('process_categories',categories)]:
        if not isinstance(value,dict): raise ValueError(f'{name} must be a JSON object.')
    if not isinstance(bounds,dict): raise ValueError('Composition bounds must be an object {element: [lower, upper]} in at.%.')
    missing=set(c['elements'])-set(bounds); unknown=set(bounds)-set(c['elements'])
    if missing or unknown: raise ValueError(f'Composition bound keys: missing={sorted(missing)}, unknown={sorted(unknown)}. Use exact selected elements and at.% units.')
    if mode not in ('fixed','observed','bounded'): raise ValueError(f'Unsupported process_mode {mode!r}; use fixed, observed or bounded.')
    for name,fields in [('conditions',cond),('process_ranges',ranges),('process_categories',categories)]:
        unknown=set(fields)-set(c['extras'])
        if unknown: raise ValueError(f'{name}: unknown or unselected process fields {sorted(unknown)}. Select them as inputs before planning.')
    overlap=(set(cond)&set(ranges))|(set(cond)&set(categories))|(set(ranges)&set(categories))
    if overlap: raise ValueError(f'Conflicting fixed/range/category constraints: {sorted(overlap)}. Specify each field only once.')
    if mode=='fixed':
        if ranges or categories: raise ValueError('Fixed mode cannot search ranges or category sets; select observed or bounded mode.')
        if set(cond)!=set(c['extras']): raise ValueError('Fixed processing conditions must match selected inputs')
    elif not c['extras'] or not ps['observed_combinations']: raise ValueError('Process search requires selected inputs and complete recorded combinations.')
    def finite(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v)
    for e,v in ranges.items():
        f=ps['fields'][e]
        if f['type']!='numeric': raise ValueError(f'{e}: categorical process field requires process_categories, not numeric bounds.')
        if not isinstance(v,(list,tuple)) or len(v)!=2 or not all(finite(x) for x in v) or v[0]>v[1]: raise ValueError(f'{e}: invalid process range {v!r}; require finite lower <= upper in dataset units.')
        if v[0]<f['min'] or v[1]>f['max']: raise ValueError(f'{e}: requested range {v} outside observed domain [{f["min"]}, {f["max"]}] in dataset units; review units/range. No automatic clipping.')
    for e,v in categories.items():
        f=ps['fields'][e]
        if f['type']!='categorical' or not isinstance(v,list) or not v or any(str(x).strip() not in f['values'] for x in v): raise ValueError(f'{e}: select a nonempty list of recorded categories; allowed: {f.get("values",[])}.')
    for e,v in bounds.items():
        try: valid=isinstance(v,(list,tuple)) and len(v)==2 and all(np.isfinite(float(x)) for x in v) and 0<=float(v[0])<=float(v[1])<=100
        except (TypeError,ValueError): valid=False
        if not valid: raise ValueError(f'{e}: invalid bounds {v!r}; require 0 <= lower <= upper <= 100 at.%.')
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v): raise ValueError(f'{e}: bounds must be JSON numbers.')
    lo=sum(v[0] for v in bounds.values()); hi=sum(v[1] for v in bounds.values())
    if lo>100+1e-8 or hi<100-1e-8: raise ValueError(f'Infeasible composition bounds: sum(lower)={lo:g}, sum(upper)={hi:g} at.%; require sum(lower) <= 100 <= sum(upper). Review other element ranges, or revise the requested constraint.')
    axes=b.get('plot_axes')
    if axes is not None:
        if not isinstance(axes,dict) or axes.get('y_target') not in dirs or ('x_element' in axes)==('x_process' in axes): raise ValueError('plot_axes requires one x_element or x_process and y_target from optimized targets.')
        if 'x_element' in axes and axes['x_element'] not in c['elements']: raise ValueError('Unknown plot element.')
        if 'x_process' in axes and (axes['x_process'] not in c['extras'] or ps['fields'][axes['x_process']]['type']!='numeric'): raise ValueError('Plot process must be a selected numerical field.')
        if 'bin_width' in axes and (not finite(axes['bin_width']) or axes['bin_width']<=0): raise ValueError('Envelope bin_width must be positive in axis units.')
    top_n=b.get('top_n',20)
    if not finite(top_n) or not 1<=top_n<=1000 or top_n!=int(top_n): raise ValueError('top_n must be an integer 1..1000.')
    for e,v in cond.items():
        f=ps['fields'][e]
        if f['type']=='categorical':
            if str(v).strip() not in f['values']: raise ValueError(f'{e}: select a recorded processing category.')
        elif not finite(v) or not f['min']<=v<=f['max']: raise ValueError(f'{e}: fixed numeric condition must lie in observed range [{f["min"]}, {f["max"]}] in dataset units.')
    if not set(limits)<=set(dirs) or not all(finite(v) for v in limits.values()): raise ValueError('Invalid performance thresholds')
    for k,lo,hi,integer in [('iterations',3,250,True),('pool_size',3,10000,True),('step',.1,20,False),('min_entropy',0,R_MAX(c),False)]:
        v=b.get(k,0 if k=='min_entropy' else None)
        if not finite(v) or not lo<=v<=hi or (integer and v!=int(v)): raise ValueError(f'{k}: require {lo:g} <= value <= {hi:g}'+(' (integer)' if integer else ''))
    pool_grid(c['elements'],bounds,float(b['step']),3,int(c['seed']))
    if mode!='fixed': constrained_process_pool(df,c['extras'],categorical_columns(df,c),mode,cond,ranges,categories)


def canonical_design(c,b,df):
    if not isinstance(b,dict): raise ValueError('design must be an object')
    f=fractions(df,c['elements'],c['unit'])*100
    bounds={e:[float(np.floor(f[:,i].min())),float(np.ceil(f[:,i].max()))] for i,e in enumerate(c['elements'])}
    return normalize_design(default_design(df,c,bounds),b,c,df)[0]

@app.post('/api/validate-design/{sid}')
def preview_design(sid:str,b:dict):
    s=session(sid)
    try:
        c=b.get('config') or s.get('config'); check(s,c)
        d=canonical_design(c,b['design'],s['df']); validate_design(c,d,s['df'])
        return {'valid':True,'design':d}
    except Exception as e: raise HTTPException(400,str(e))

def R_MAX(c): return 8.314462618*np.log(len(c['elements']))+1e-6

def design_stage(s,b):
    c=s['config']; directions=b['directions']; log(s,'STAGE 3/3 · Bayesian optimization and full-pool screening')
    for key in ('single_ranking','top_recommendations','optimal_envelope'): s.pop(key,None)
    for key in list(s['plots']):
        if key.startswith(('pareto_','single_','envelope_')): s['plots'].pop(key)
    for key in list(s['plot_files']):
        if key.startswith(('pareto_','single_','envelope_')): s['plot_files'].pop(key)
    cond=b.get('conditions',{}); out,hist,n=inverse(s['df'],s['results'],c['elements'],c['descriptors'],c['extras'],c['raw'],b['bounds'],cond,directions,float(b['step']),int(b['pool_size']),int(b['iterations']),int(c['seed']),float(b.get('min_entropy',0)),b.get('thresholds',{}),categorical=c['categorical_extras'],process_mode=b.get('process_mode','fixed'),source_unit=c['unit'],process_ranges=b.get('process_ranges',{}),process_categories=b.get('process_categories',{}))
    s.update(design=out,history=hist,design_config=b); tg=list(directions); p=out[out.Pareto]; from itertools import combinations
    pairs=list(combinations(tg,2))
    for i,(a,z) in enumerate(pairs):
        fig,ax=plt.subplots(figsize=(6.4,5.4)); x='pred_'+a; y='pred_'+z if z else 'S_mix_J_molK'; f=out[out.Feasible]; nf=out[~out.Feasible]
        if len(nf): ax.scatter(nf[x],nf[y],color='#C8CBD0',marker='x',s=35,label='Threshold violation')
        ax.scatter(f[x],f[y],color=COLORS[0],alpha=.65,s=43,label='Feasible evaluated'); ax.scatter(p[x],p[y],color=COLORS[1],s=70,edgecolors='white',label='Pareto'); ax.set(xlabel=a,ylabel=z or 'Configurational entropy (J/mol K)'); ax.legend(fontsize=10); save_plot(s,f'pareto_{i+1}',fig)
    axes=b.get('plot_axes')
    if axes and len(tg)>1:
        e=axes.get('x_element') or axes.get('x_process'); t=axes['y_target']; y='pred_'+t
        fig,ax=plt.subplots(figsize=(6.4,5.4)); f=out[out.Feasible]; nf=out[~out.Feasible]
        if len(nf): ax.scatter(nf[e],nf[y],color='#C8CBD0',marker='x',s=35,label='Threshold violation')
        ax.scatter(f[e],f[y],color=COLORS[0],alpha=.65,s=43,label='Feasible evaluated')
        ax.scatter(p[e],p[y],color=COLORS[1],s=70,edgecolors='white',label='Pareto objectives' if len(tg)>1 else 'Best predicted objective')
        ax.set(xlabel=e+(' (at.%)' if 'x_element' in axes else ''),ylabel='Predicted '+t); ax.legend(fontsize=10); save_plot(s,'pareto_composition_property',fig)
    extra={}
    if len(tg)==1:
        t=tg[0]; direction=directions[t]; y='pred_'+t
        ranking,top,envelope=single_results(out,t,direction,int(b.get('top_n',20)),axes)
        s.update(single_ranking=ranking,top_recommendations=top,optimal_envelope=envelope)
        extra=dict(mode='single',top_n_requested=int(b.get('top_n',20)),top_recommendations=jsonable(top),envelope=jsonable(envelope),envelope_axes=axes)
        if len(ranking):
            fig,ax=plt.subplots(figsize=(6.4,5.0)); ax.plot(ranking.objective_rank,ranking[y],color=COLORS[0],lw=1.5,label='Feasible ranking'); ax.scatter(top.objective_rank,top[y],color=COLORS[1],s=25,label='Top-N'); ax.set(xlabel='Objective rank',ylabel='Predicted '+t); ax.legend(fontsize=10); save_plot(s,'single_objective_ranking',fig)
        if axes:
            e=axes.get('x_element') or axes.get('x_process'); fig,ax=plt.subplots(figsize=(6.4,5.4)); f=out[out.Feasible]; nf=out[~out.Feasible]
            if len(nf): ax.scatter(nf[e],nf[y],color='#C8CBD0',marker='x',s=25,label='Threshold violation')
            if len(f): ax.scatter(f[e],f[y],color=COLORS[0],alpha=.45,s=30,label='Feasible candidates')
            if len(envelope): ax.plot(envelope[e],envelope[y],'-o',color=COLORS[1],lw=1.8,ms=5,label='Conditional optimal envelope')
            if len(p): ax.scatter(p[e],p[y],color=COLORS[2],s=80,edgecolors='white',label='Best in pool')
            ax.set(xlabel=e+(' (at.%)' if 'x_element' in axes else ''),ylabel='Predicted '+t); ax.legend(fontsize=10); save_plot(s,'envelope_single_objective',fig)
        # Preserve the mathematical Pareto flag; rank feasible candidates separately.
        ordered=out.merge(ranking[['pool_index','objective_rank']],on='pool_index',how='left').sort_values(['Feasible','objective_rank'],ascending=[False,True],na_position='last')
        s['design']=ordered
    else:
        extra=dict(mode='multi'); ordered=out.sort_values(['Pareto','compromise_score'],ascending=[False,True],na_position='last')
    s['inverse_summary']=dict(pool=n,evaluated=len(out),bo_evaluated=len(hist),batch_evaluated=n-len(hist),pareto_scope='full_candidate_pool',feasible=int(out.Feasible.sum()),pareto=len(p),objectives=directions,process_mode=b.get('process_mode','fixed'),thresholds=b.get('thresholds',{}),process_ranges=b.get('process_ranges',{}),process_categories=b.get('process_categories',{}),rows=jsonable(ordered),figures=[k for k in s['plots'] if k.startswith(('pareto_','single_','envelope_'))],**extra); log(s,f'候选池 {n}，评估 {len(out)}，非支配解 {len(p)}'); log(s,f'Full-pool evaluation: BO={len(hist)}, additional batch={n-len(hist)}, total={len(out)}')

@app.post('/api/inverse/{sid}')
def design(sid:str,b:dict):
    s=session(sid)
    if not s.get('results') or not s.get('summary'): raise HTTPException(400,'请先完成训练。')
    try:
        b=canonical_design(s['config'],b,s['df']); validate_design(s['config'],b,s['df'])
    except Exception as e: raise HTTPException(400,str(e))
    return start(s,lambda:design_stage(s,b))

@app.post('/api/workflow/{sid}')
def workflow(sid:str,b:dict):
    s=session(sid)
    try:
        if b.get('approved') is not True or b.get('signature')!=s['signature']: raise ValueError('Explicit approval and matching dataset signature required')
        c=b['config']; d=b.get('design'); validate_train(s,c)
        lp=s.get('llm_plan',{})
        if lp.get('provider') not in (None,'manual','adaptive_local'):
            if b.get('plan_id')!=lp.get('plan_id') or lp.get('planning_stage')!=2: raise ValueError('Stage 2 planning required before LLM workflow approval / 请先完成第二阶段大模型规划。')
            if not c.get('target_settings') or feature_signature(c)!=lp.get('feature_signature'): raise ValueError('Selected feature configuration changed; rerun stage 2 / 特征配置已改变，请重新执行第二阶段规划。')
        if c.get('target_settings'):
            if not b.get('plan_id') or s.get('llm_plan',{}).get('plan_id')!=b.get('plan_id'): raise ValueError('Adaptive plan ID is missing or stale. Generate a new plan before approval.')
            planned=s['llm_plan']['config']
            if any(c.get(k)!=planned.get(k) for k in ('test','seed','group')): raise ValueError('Adaptive split settings changed. Regenerate planning before approval.')
        if d is not None:
            d=canonical_design(c,d,s['df']); validate_design(c,d,s['df'])
    except Exception as e: raise HTTPException(400,str(e))
    def work():
        s['approved_workflow']=dict(config=c,design=d,signature=s['signature'],plan_id=b.get('plan_id'),approved_at=time.strftime('%Y-%m-%d %H:%M:%S')); log(s,'Human-approved workflow started'); train_stage(s,c)
        if d is not None: design_stage(s,d)
        log(s,'Workflow finished · results ready for human review')
    return start(s,work)
@app.get('/api/plots/{sid}')
def plots(sid:str): return session(sid)['plots']
from llm import request_plan, default_design
from design_planning import normalize_design

def planning_config(s,c,language):
    c=dict(c); df=s['df']; warnings=[]; els=c.get('elements',[])
    for field,info in column_profile(df,[e for e in df if e in ELEMENTS]).items():
        if field in c.get('extras',[]) and info.get('warning'):
            warnings.append(f'{field}: {info["warning"]}')
    if not els:
        els=[e for e in df if e in ELEMENTS]; c['elements']=els
        if els: warnings.append('Automatically selected detected element columns: '+', '.join(els) if language=='en' else '已自动选择识别出的元素列：'+', '.join(els))
    if not els or not set(els)<=set(ELEMENTS) or not set(els)<=set(df.columns):
        raise ValueError('No valid element columns detected / 未识别到有效元素列。请使用 Fe、Ni、Fe(at.%)、铁含量等独立数值列；当前表头 / headers: '+', '.join(df.columns[:25]))
    hint,provisional=proposed_unit(df,els,s.get('unit_hints',{}))
    if c.get('unit') in ('auto',None,''):
        if hint is None: raise ValueError('Cannot infer composition scale / 无法判断成分单位，请检查元素列、缺失值、负数及行合计（1 或 100），再手动指定单位。')
        c['unit']=hint
        if provisional:
            warnings.append(('PROVISIONAL UNIT: '+hint+'. Row totals cannot distinguish atomic and weight percentages. Verify source units before approval; if different, select actual units in Data Import and regenerate the plan.' if language=='en' else '待审核单位：'+hint+'。行合计无法区分原子百分比与质量百分比。审批前务必核对原始数据；如不同，请在数据导入页选择真实单位并重新生成方案。'))
    else:
        if not provisional and hint and c['unit']!=hint: raise ValueError('Selected unit conflicts with headers / 所选单位与表头标注冲突。')
    fractions(df,els,c['unit'])
    return c,warnings
@app.post('/api/adaptive-plan/{sid}')
def adaptive_plan(sid:str,b:dict):
    s=session(sid)
    if s['status']=='running': raise HTTPException(409,'A workflow is running')
    try:
        c,warnings=planning_config(s,b['config'],b.get('language')); df=s['df']; els=c['elements']
        if not c.get('targets'): c['targets']=[k for k,v in column_profile(df,els).items() if v['role']=='target' and v['numeric_count']>=12]
        if not c.get('targets'): raise ValueError('Select at least one numeric target with >=12 samples.')
        base,profiles,warning=baseline(df,c)
        if warning: warnings.append(warning)
        validate_train(s,base)
        d=None
        if b.get('inverse') is True:
            f=fractions(df,els,base['unit'])*100
            bounds={e:[float(np.floor(f[:,i].min())),float(np.ceil(f[:,i].max()))] for i,e in enumerate(els)}
            d=default_design(df,base,bounds); dim=len(els)+len(base['extras']); d.update(pool_size=min(10000,max(300,300*dim)),iterations=min(250,max(30,8*dim+5*len(base['targets']))))
            validate_design(base,d,df)
        result=dict(version=VERSION,plan_id=uuid.uuid4().hex,signature=s['signature'],provider='adaptive_local',rationale=('Training-data adaptive plan; no external LLM was called. Review objectives and constraints before approval.' if b.get('language')=='en' else '基于训练集实际统计的自适应方案，未调用外部大模型。请审核目标、算法与约束；本地规则不会解析自然语言研究目标。'),config=base,design=d,mapping_warnings=warnings,training_profiles=jsonable(profiles),adaptive_baseline=base,setting_sources={t:{k:'adaptive_data_rule' for k in v} for t,v in base['target_settings'].items()},input_config=c)
        s['llm_plan']=result; log(s,'Data-adaptive local plan generated · awaiting human approval'); return result
    except Exception as e: raise HTTPException(400,'Adaptive planning failed: '+str(e)[:500])

@app.post('/api/local-plan/{sid}')
def local_plan(sid:str,b:dict):
    s=session(sid)
    try:
        c,warnings=planning_config(s,b['config'],b.get('language')); validate_train(s,c); f=fractions(s['df'],c['elements'],c['unit'])*100
        bounds={e:[float(np.floor(f[:,i].min())),float(np.ceil(f[:,i].max()))] for i,e in enumerate(c['elements'])}
        d=default_design(s['df'],c,bounds); validate_design(c,d,s['df'])
        p=dict(plan_id=uuid.uuid4().hex,signature=s['signature'],provider='manual',rationale='Manual plan: review and edit objective directions, constraints and budget.' if b.get('language')=='en' else '手动方案：请审核并调整目标方向、约束和预算。',config=c,design=d,mapping_warnings=warnings); s['llm_plan']=p; return p
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/api/plan/{sid}')
def plan(sid:str,b:dict):
    s=session(sid)
    if s['status']=='running': raise HTTPException(409,'A workflow is running')
    try:
        c,warnings=planning_config(s,b['config'],b.get('language')); df=s['df']; els=c['elements']
        f=fractions(df,els,c['unit'])*100
        if not c.get('targets'): c['targets']=[k for k,v in column_profile(df,els).items() if v['role']=='target' and v['numeric_count']>=12]
        if not c.get('targets'): raise ValueError('Select at least one numeric target with >=12 samples.')
        c['categorical_extras']=categorical_columns(df,c)
        base,profiles,warning=baseline(df,c)
        if warning: warnings.append(warning)
        cp=column_profile(df,els)
        cp={k:{'role':v['role'],'type':v['type']} for k,v in cp.items()}
        bounds={e:[float(np.floor(f[:,i].min())),float(np.ceil(f[:,i].max()))] for i,e in enumerate(els)}
        profile=dict(columns=list(df.columns),column_profile=cp,training_profiles=jsonable(profiles),adaptive_baseline=base,composition_ranges=bounds,design_process_options=process_summary(df,base),split_policy='test fraction, group and seed fixed before target profiling; per-target folds may vary. Design ranges are full-data input-domain metadata, not target statistics.')
        registry=dict(descriptors=DESCRIPTORS,selectors=SELECTORS,models=list(model_specs()),optimizer=OPTIMIZERS)
        obj=request_plan(b,profile,registry)
        suggested=obj['config']; merged,sources,notices=merge_adaptive(df,base,profiles,suggested); warnings.extend(notices)
        omitted=[f'{t}: '+', '.join(k for k,v in q.items() if v=='adaptive_data_rule') for t,q in sources.items() if 'adaptive_data_rule' in q.values()]
        if omitted: warnings.append('LLM omitted settings; data-adaptive rules supplied them (not UI defaults): '+ '; '.join(omitted))
        validate_train(s,merged)
        d=None; design_errors=[]; design_sources={}
        if obj.get('design') is not None:
            d=default_design(df,merged,profile['composition_ranges']); dimensions=len(els)+len(merged['extras']); nobj=len(obj['design']['directions']) if isinstance(obj['design'].get('directions'),dict) else len(merged['targets']); d.update(pool_size=min(10000,max(300,300*dimensions)),iterations=min(250,max(30,8*dimensions+5*nobj))); d,notes,design_sources=normalize_design(d,obj['design'],merged,df); warnings.extend(notes)
            for key in ('pool_size','iterations'):
                if key not in obj['design']: design_sources[key]='adaptive_data_rule'
            try: validate_design(merged,d,df)
            except (ValueError,TypeError,KeyError,AttributeError) as ex:
                design_errors.append(str(ex)); warnings.append('DESIGN REVIEW REQUIRED / 逆向设计待修正: '+str(ex)+'; edit the design JSON before execution. No requested bounds were widened.')
        result=dict(version=VERSION,input_config=c,plan_id=uuid.uuid4().hex,signature=s['signature'],provider=b.get('provider','openai'),planning_stage=1,model=b['model'],goal=b.get('goal',''),rationale=str(obj.get('rationale','')),config=merged,design=d,mapping_warnings=warnings,training_profiles=jsonable(profiles),adaptive_baseline=base,setting_sources=sources,common_setting_sources={k:('LLM' if k in suggested else 'adaptive_data_rule') for k in ('descriptors','raw')},design_sources=design_sources,design_errors=design_errors)
        s['llm_plan']=result; log(s,'LLM stage 1 generated · awaiting feature selection and stage 2'); return result
    except Exception as e: raise HTTPException(400,'Planning failed: '+str(e)[:500])
@app.post('/api/plan-stage2/{sid}')
def plan_stage2(sid:str,b:dict):
    s=session(sid)
    if s['status']=='running': raise HTTPException(409,'A workflow is running')
    try:
        first=s.get('llm_plan',{})
        if b.get('plan_id')!=first.get('plan_id') or first.get('provider') in (None,'manual','adaptive_local'): raise ValueError('Generate an LLM stage 1 plan first / 请先生成第一阶段大模型方案。')
        if b.get('provider','openai')!=first['provider'] or b.get('model')!=first['model']: raise ValueError('Provider or model changed; regenerate stage 1.')
        c=copy.deepcopy(b['config']); validate_train(s,c)
        if not c.get('target_settings'): raise ValueError('Per-target settings required')
        if any(c.get(k)!=first['config'].get(k) for k in ('test','seed','group','elements','unit')): raise ValueError('Frozen data mapping or split changed; regenerate stage 1.')
        if not set(c['targets'])<=set(first['config']['targets']): raise ValueError('Target changed; regenerate stage 1.')
        d=copy.deepcopy(b.get('design'))
        # Keep design review independent: stage 2 cannot mutate constraints.
        profiles,rankings=selected_profiles(s['df'],get_features(s,c),c); base=training_baseline(c,profiles)
        registry=dict(models=list(model_specs()),optimizer=OPTIMIZERS)
        profile=dict(selected_training_profiles=jsonable(profiles),training_baseline={t:{k:v[k] for k in ('models','optimizer','budget','folds','screen_top')} for t,v in base['target_settings'].items()},split_policy='Training-only statistics; holdout untouched; CV refits feature selection within folds.')
        log(s,'LLM stage 2 · actual selected feature diagnostics computed')
        obj=request_plan(dict(b,stage=2,goal=first.get('goal','')),profile,registry)
        if obj.get('design') is not None: raise ValueError('Stage 2 cannot modify design; return design:null.')
        merged,sources,warnings=merge_stage_two(base,obj['config']); validate_train(s,merged)
        if feature_signature(merged)!=feature_signature(c): raise ValueError('Stage 2 changed selected features')
        result=copy.deepcopy(first); result.update(plan_id=uuid.uuid4().hex,planning_stage=2,feature_signature=feature_signature(c),stage1_plan_id=first.get('stage1_plan_id',first['plan_id']),stage1_config=first.get('stage1_config',first['config']),stage1_rationale=first.get('stage1_rationale',first['rationale']),stage2_rationale=str(obj.get('rationale','')),config=merged,design=d,selected_training_profiles=jsonable(profiles),selected_rankings=jsonable(rankings),stage2_baseline=base,mapping_warnings=list(first.get('mapping_warnings',[]))+warnings)
        result['rationale']=result['stage1_rationale']+'\n\nSTAGE 2 · '+result['stage2_rationale']; result['setting_sources']={t:{**first.get('setting_sources',{}).get(t,{}),**sources[t]} for t in merged['targets']}
        s['llm_plan']=result; log(s,'LLM stage 2 completed · awaiting final human approval'); return jsonable(result)
    except Exception as e: raise HTTPException(400,'Stage 2 planning failed: '+str(e)[:600])

@app.post('/api/literature')
def literature(b:dict):
    # Local PDF/text extraction; provenance is page based, no invented values.
    try:
        data=base64.b64decode(b['data']); name=b['name'];
        if len(data)>30*1024**2: raise ValueError('最大30MB。')
        if name.lower().endswith('.pdf'):
            from pypdf import PdfReader
            pages=[{'page':i+1,'text':p.extract_text() or ''} for i,p in enumerate(PdfReader(io.BytesIO(data)).pages)]
        elif name.lower().endswith('.txt'): pages=[{'page':1,'text':data.decode('utf-8-sig')}]
        else: raise ValueError('支持 PDF 或 TXT。')
        return dict(name=name,pages=pages,sha256=hashlib.sha256(data).hexdigest())
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/export/{sid}')
def export(sid:str):
    s=session(sid)
    if s['status']=='running': raise HTTPException(409,'任务尚未完成。')
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('source.csv',s.get('original_df',s['df']).to_csv(index=False,encoding='utf-8')); z.writestr('normalized_headers.csv',s['df'].to_csv(index=False,encoding='utf-8')); z.writestr('column_mapping.json',json.dumps({'header_mapping':s.get('header_mapping',{}),'unit_hints':s.get('unit_hints',{})},ensure_ascii=False,indent=2)); z.writestr('provenance.json',json.dumps(jsonable({k:s.get(k) for k in ('filename','signature','config','feature_config','design_config','llm_plan','approved_workflow','logs','summary','inverse_summary')}),ensure_ascii=False,indent=2));
        for k in ('feature_table','comparison','failures','traces','design','history','single_ranking','top_recommendations','optimal_envelope'):
            if k in s: z.writestr(k+'.csv',s[k].to_csv(index=False))
        for i,(t,r) in enumerate(s.get('results',{}).items()):
            z.writestr(f'models/{i+1}_holdout.csv',r['holdout'].to_csv(index=False)); z.writestr(f'models/{i+1}_ranking.csv',r['ranking'].to_csv(index=False)); z.writestr(f'models/{i+1}_split.json',json.dumps(jsonable({'target':t,'train_rows':r['train_rows'],'test_rows':r['test_rows'],'features':r['cols'],'encoded_features':r['encoded_cols']}))); f=io.BytesIO(); joblib.dump(r['model'],f); z.writestr(f'models/{i+1}.joblib',f.getvalue())
        for k,v in s['plot_files'].items(): z.writestr('figures/'+k,v)
    buff.seek(0); return StreamingResponse(buff,media_type='application/zip',headers={'Content-Disposition':'attachment; filename=EntropyForge_results.zip'})
