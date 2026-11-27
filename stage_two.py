"""Stage-two planning: only the actual selected training feature space is profiled."""
import copy, hashlib, json
import numpy as np
import pandas as pd
from engine import split_target, feature_preprocessor, SelectFeatures, feature_ranking
from adaptive import recommend, summarize, validate_target_settings

TRAIN_KEYS=('models','optimizer','budget','folds','screen_top')

def feature_signature(c):
    fields={k:c.get(k) for k in ('elements','unit','targets','extras','categorical_extras','descriptors','raw','test','seed','group')}
    fields['selection']={t:{k:(c.get('target_settings') or {}).get(t,c)[k] for k in ('selectors','k')} for t in c['targets']}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def selected_profiles(df,X,c):
    profiles={}; rankings={}
    for t in c['targets']:
        v=c['target_settings'][t]; idx,tr,te,cv=split_target(df,t,c.get('group'),float(c['test']),int(c['seed']),int(v['folds']))
        xx=X.iloc[idx].reset_index(drop=True); xx=xx.loc[:,xx.iloc[tr].notna().any()]; tx=xx.iloc[tr]; y=pd.to_numeric(df[t],errors='coerce').iloc[idx].to_numpy(float)[tr]
        if not len(tx.columns): raise ValueError(f'{t}: no usable training features')
        pre=feature_preprocessor(xx); encoded=pre.fit_transform(tx); sel=SelectFeatures(tuple(v['selectors']),int(v['k']),int(c['seed'])).fit(encoded,y)
        rank=feature_ranking(pre,sel); rankings[t]=rank
        names=pre.get_feature_names_out()[sel.indices_].tolist(); a=pd.DataFrame(encoded[:,sel.indices_],columns=names); yy=pd.Series(y)
        pear=a.corrwith(yy).abs().dropna(); spear=a.corrwith(yy,method='spearman').abs().dropna(); gap=(spear-pear).dropna(); corr=a.corr().abs().to_numpy(); pairs=corr[np.triu_indices(len(names),1)]; pairs=pairs[np.isfinite(pairs)]
        q=np.quantile(y,[0,.25,.5,.75,1]); iqr=q[3]-q[1]; original_missing=tx.isna().mean()
        # Encoded names are used exactly as in the fitted feature selector.
        profiles[t]=dict(n_train=len(tr),n_test=len(te),effective_inputs=len(names),requested_k=v['k'],available=encoded.shape[1],selected_features=names,selected_scores=rank.loc[rank.selected,['feature','score']].to_dict('records'),sample_feature_ratio=len(tr)/len(names),constant_selected=int((a.nunique()<=1).sum()),categorical_levels={name:int(a[name].nunique()) for name in names if name.startswith('cat__')},missing_fraction=float(original_missing.mean()),missingness_scope='before encoding, all available training inputs',max_abs_pearson=float(pear.max()) if len(pear) else 0.,max_abs_spearman=float(spear.max()) if len(spear) else 0.,rank_linear_gap=float(max(0,gap.max())) if len(gap) else 0.,selected_feature_pearson={str(k):float(x) for k,x in pear.items()},selected_feature_spearman={str(k):float(x) for k,x in spear.items()},mean_abs_interfeature_corr=float(pairs.mean()) if len(pairs) else 0.,high_corr_pair_fraction=float((pairs>=.9).mean()) if len(pairs) else 0.,target_std=float(np.std(y,ddof=1)),target_iqr=float(iqr),target_outlier_fraction=float(((y<q[1]-1.5*iqr)|(y>q[3]+1.5*iqr)).mean()) if iqr>0 else 0.,target_quantiles=q.tolist(),cv_folds=len(cv),min_cv_train=min(len(a) for a,b in cv))
    return profiles,rankings

def training_baseline(c,profiles):
    base=copy.deepcopy(c)
    for t,p in profiles.items():
        rec=recommend(p,0)
        for key in TRAIN_KEYS: base['target_settings'][t][key]=rec[key]
    summarize(base); validate_target_settings(base); return base

def merge_stage_two(base,suggested):
    if not isinstance(suggested,dict) or set(suggested)-{'target_settings'}: raise ValueError('Stage 2 may only change per-target training settings, not selected features, splits or design.')
    per=suggested.get('target_settings',{})
    if not isinstance(per,dict) or set(per)-set(base['targets']): raise ValueError('Invalid stage 2 target keys')
    c=copy.deepcopy(base); sources={}; warnings=[]
    for t in c['targets']:
        v=per.get(t,{})
        if not isinstance(v,dict) or set(v)-set(TRAIN_KEYS): raise ValueError(f'{t}: stage 2 may only set {TRAIN_KEYS}')
        cfg=c['target_settings'][t]; cfg.update(v)
        if 'screen_top' not in v: cfg['screen_top']=min(cfg['screen_top'],len(cfg['models']))
        sources[t]={k:('LLM_stage2' if k in v else 'selected_feature_data_rule') for k in TRAIN_KEYS}
        omitted=[k for k in TRAIN_KEYS if k not in v]
        if omitted: warnings.append(f'{t}: stage 2 omitted {", ".join(omitted)}; selected-feature adaptive rules supplied these settings.')
    summarize(c); validate_target_settings(c); return c,sources,warnings
