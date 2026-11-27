"""Dataset-dependent planning. Profiles and numerical decisions use training rows only."""
import numpy as np
import pandas as pd
from engine import DESCRIPTORS, model_specs, SELECTORS, OPTIMIZERS, fractions, split_target, descriptors
from process import process_frame, categorical_columns

TARGET_KEYS=('models','selectors','k','optimizer','budget','folds','screen_top')

def profile_target(df,c,target):
    idx,tr,te,cv=split_target(df,target,c.get('group'),c['test'],c['seed'],c['folds'])
    a=df.iloc[idx[tr]].reset_index(drop=True); y=pd.to_numeric(a[target],errors='coerce'); x=pd.DataFrame(fractions(a,c['elements'],c['unit'])*100,columns=c['elements']); ex=process_frame(a,c['extras'],categorical_columns(df,c)); x=pd.concat([x,ex],axis=1)
    numeric=x.select_dtypes('number'); pear=numeric.corrwith(y).abs().dropna(); spear=numeric.corrwith(y,method='spearman').abs().dropna(); delta=(spear-pear).dropna();
    p=sum(int(x[e].nunique(dropna=True)>1) if e in numeric else int(x[e].nunique(dropna=True)) for e in x)
    cat={e:int(x[e].nunique()) for e in x if e not in numeric}; n=len(a); q=y.quantile([.25,.5,.75]); iqr=q.loc[.75]-q.loc[.25]; outlier=float(((y<q.loc[.25]-1.5*iqr)|(y>q.loc[.75]+1.5*iqr)).mean()) if iqr>0 else 0.
    return dict(n_train=n,n_test=len(te),effective_inputs=max(1,p),categorical_levels=cat,missing_fraction=float(x.isna().mean().mean()),target_std=float(y.std()),target_iqr=float(iqr),target_outlier_fraction=outlier,max_abs_pearson=float(pear.max()) if len(pear) else 0.,max_abs_spearman=float(spear.max()) if len(spear) else 0.,rank_linear_gap=float(max(0,delta.max())) if len(delta) else 0.,constant_target=bool(y.nunique()<2),min_cv_train=min(len(a) for a,b in cv),cv_folds=len(cv),train_groups=int(a[c['group']].nunique()) if c.get('group') else None, target_quantiles=[float(v) for v in y.quantile([0,.25,.5,.75,1])], input_missing={e:float(x[e].isna().mean()) for e in x})

def recommend(p, desc_count):
    n=p['n_train']; dim=p['effective_inputs']+desc_count; ratio=n/max(dim,1); cat=bool(p['categorical_levels']); nonlinear=p['rank_linear_gap']>.10 or p['max_abs_pearson']<.45; noisy=p['target_outlier_fraction']>.06
    if n<80:
        models=['Ridge','Bayesian ridge','Extra trees','SVR RBF','Gaussian process']; methods=['Spearman','Lasso']; budget=5; opt='Simulated annealing'; top=2
    elif n<400:
        models=['Ridge','Extra trees','Random forest','Gradient boosting','SVR RBF','Kernel ridge','XGBoost']; methods=['Spearman','Mutual information','Extra trees importance']; budget=9; opt='Particle swarm'; top=3
    elif n<2000:
        models=['Ridge','Extra trees','Random forest','Gradient boosting','Hist gradient boosting','XGBoost','LightGBM','CatBoost','SVR RBF']; methods=['Mutual information','Extra trees importance','Spearman']; budget=14; opt='Genetic algorithm'; top=4
    else:
        models=['Ridge','Extra trees','Random forest','Hist gradient boosting','XGBoost','LightGBM','CatBoost','MLP','SGD','Gradient boosting']; methods=['F test','Mutual information','Extra trees importance']; budget=18; opt='Genetic algorithm'; top=4
    if cat and n>=80 and 'CatBoost' not in models: models.append('CatBoost')
    if noisy: models.append('Huber')
    if not nonlinear and ratio<10 and 'Lasso' not in models: models.append('Lasso')
    if nonlinear and 'Extra trees importance' not in methods and n>=40: methods.append('Extra trees importance')
    k=max(1,min(100,dim,max(3,int(n/7))))
    if dim>1: k=min(k,dim-1) # default reduces dimensionality; human may explicitly retain all after ranking
    return dict(models=list(dict.fromkeys(models)),selectors=methods,k=k,optimizer=opt,budget=budget,folds=p['cv_folds'],screen_top=min(top,len(models)))

def baseline(df,c):
    c=dict(c); c['categorical_extras']=categorical_columns(df,c)
    # Freeze split settings before profiling; user can choose seed/group in the data page.
    # Use selected target counts to choose one shared holdout fraction, not target values.
    counts=[int(pd.to_numeric(df[t],errors='coerce').replace([np.inf,-np.inf],np.nan).notna().sum()) for t in c['targets']]
    n=min(counts); c['test']=.2 if n<500 else .25; c['folds']=3 if n<80 else 5
    profiles={t:profile_target(df,c,t) for t in c['targets']}
    # Descriptors are computed on training rows; retain usable nonconstant columns only.
    rows=None
    for t in c['targets']:
        ix,tr,_,_=split_target(df,t,c.get('group'),c['test'],c['seed'],c['folds']); r=set(ix[tr].tolist()); rows=r if rows is None else rows & r
    a=df.iloc[sorted(rows)]; x=fractions(a,c['elements'],c['unit']) if len(a) else None; base=DESCRIPTORS[:4]
    preferred=base+['delta_r_pct','VEC_mean','VEC_std','chi_mean','chi_std','Tm_mean_K','H_mix_kJ_mol','Omega','mass_mean','GSvolume_mean_A3']
    requested=preferred if min(p['n_train'] for p in profiles.values())<400 else DESCRIPTORS
    warning=None
    try: d=descriptors(x,c['elements'],requested) if x is not None else pd.DataFrame(columns=base)
    except ImportError:
        d=descriptors(x,c['elements'],base); warning='External elemental database unavailable; adaptive baseline uses composition-only descriptors. Install matminer/pymatgen for full coverage.'
    chosen=[k for k in d if d[k].notna().mean()>=.8 and d[k].nunique()>1]; c['descriptors']=chosen; c['raw']=True
    c['target_settings']={t:recommend(p,len(chosen)) for t,p in profiles.items()}
    # Global fields are a union/summary for manual controls; actual execution is per target.
    summarize(c)
    return c,profiles,warning

def summarize(c):
    settings=c['target_settings']; c['models']=list(dict.fromkeys(m for v in settings.values() for m in v['models'])); c['selectors']=list(dict.fromkeys(m for v in settings.values() for m in v['selectors'])); c['k']=max(v['k'] for v in settings.values()); c['budget']=max(v['budget'] for v in settings.values()); c['optimizer']=next(iter(settings.values()))['optimizer']

def merge_adaptive(df,base,profiles,suggested):
    c=dict(base); c['target_settings']={t:dict(v) for t,v in base['target_settings'].items()}; sources={t:{k:'adaptive_data_rule' for k in TARGET_KEYS} for t in c['targets']}; warnings=[]
    # Keep split frozen. A changed target set may be a subset of the profiled set.
    for k in ('test','seed','group','folds'):
        if k in suggested and suggested[k]!=base.get(k): warnings.append(f'{k}: frozen for training-only profiling; edit before generating a new plan if needed.')
    for k in ('targets','extras','categorical_extras','descriptors','raw'):
        if k in suggested: c[k]=suggested[k]
    if not c['targets'] or not set(c['targets'])<=set(profiles): raise ValueError('LLM targets must be among profiled targets; select additional targets in Data Import then regenerate.')
    if 'extras' in suggested and 'categorical_extras' not in suggested: c.pop('categorical_extras',None)
    c['categorical_extras']=categorical_columns(df,c)
    per=suggested.get('target_settings',{});
    if not isinstance(per,dict) or not set(per)<=set(c['targets']): raise ValueError('Invalid target_settings keys')
    c['target_settings']={t:c['target_settings'][t] for t in c['targets']}; sources={t:sources[t] for t in c['targets']}
    for t,v in c['target_settings'].items():
        specific=per.get(t,{})
        if not isinstance(specific,dict) or not set(specific)<=set(TARGET_KEYS): raise ValueError(f'{t}: invalid per-target settings')
        for k in TARGET_KEYS:
            if k in specific: v[k]=specific[k]; sources[t][k]='LLM_target'
            elif k in suggested and k!='folds': v[k]=suggested[k]; sources[t][k]='LLM_common'
        if v['selectors']==[]:
            v['selectors']=list(base['target_settings'][t]['selectors']); sources[t]['selectors']='mandatory_adaptive_rule'; warnings.append(f'{t}: empty LLM selectors replaced by mandatory data-adaptive evaluation methods.')
        if 'screen_top' not in specific and 'screen_top' not in suggested: v['screen_top']=min(v['screen_top'],len(v['models']))
        if v['optimizer']=='None': v['screen_top']=len(v['models'])
    summarize(c)
    return c,sources,warnings

def validate_target_settings(c):
    v=c.get('target_settings')
    if v is None: return
    if not isinstance(v,dict) or set(v)!=set(c['targets']): raise ValueError('Per-target configuration must match target list')
    for t,a in v.items():
        if set(a)!=set(TARGET_KEYS) or not isinstance(a['models'],list) or not a['models'] or len(set(a['models']))!=len(a['models']) or not set(a['models'])<=set(model_specs()) or not isinstance(a['selectors'],list) or not a['selectors'] or len(set(a['selectors']))!=len(a['selectors']) or not set(a['selectors'])<=set(SELECTORS) or a['optimizer'] not in OPTIMIZERS: raise ValueError(f'{t}: invalid per-target algorithm settings')
        for key,low,high in [('k',1,100),('budget',1,100),('folds',2,10),('screen_top',1,len(a['models']))]:
            if type(a[key]) is not int or not low<=a[key]<=high: raise ValueError(f'{t}: {key} outside valid range')
