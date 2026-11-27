"""Column-role suggestions and explicit process typing. Suggestions require user review."""
import re
import numpy as np
import pandas as pd
from mapping import ambiguous_cr, bare_uppercase_alias, identify

TARGET_RE=re.compile(r'^(?:ys|uts|el|hv|kic|hardness|yield[ _-]*strength|ultimate[ _-]*tensile[ _-]*strength|tensile[ _-]*strength|elongation|fracture[ _-]*toughness)(?:\b|_|\d)|屈服|抗拉|延伸|伸长|硬度|断裂韧性',re.I)
META_RE=re.compile(r'^(?:id|index|sample[ _-]*id|alloy[ _-]*id|reference|doi|source|author|paper|citation|notes?|remarks?)(?:\b|_)|编号|文献|来源|备注|作者',re.I)
NUM_RE=re.compile(r'temp|duration|time|rate|pressure|speed|reduction|thickness|strain|stress|power|cycle|frequency|energy|current|voltage|温度|时间|速率|速度|压力|变形量|压下|厚度|应变|功率|电流|电压|次数',re.I)
CAT_RE=re.compile(r'process|processing|manufactur|route|condition|state|treatment|cast|anneal|rolling|rolled|quench|aging|ageing|sinter|forg|extrud|milling|cooling|工艺|制备|状态|铸|退火|轧|淬|时效|烧结|锻|挤压|球磨|冷却|热处理',re.I)

def clean_text(s):
    return s.map(lambda x: np.nan if pd.isna(x) or not str(x).strip() else str(x).strip())

def numeric(s):
    return pd.to_numeric(clean_text(s),errors='coerce').replace([np.inf,-np.inf],np.nan)

def process_type(s,name):
    a=clean_text(s); observed=a.notna(); n=int(observed.sum())
    # Integer labels in process/state columns are nominal, not ordinal.
    if CAT_RE.search(name) and not NUM_RE.search(name): return 'categorical'
    return 'numeric' if n and numeric(a).notna().sum()==n else 'categorical'

def column_profile(df,elements):
    out={}
    for name in df:
        s=clean_text(df[name]); v=numeric(s); typ=process_type(df[name],name)
        if name in elements: role='element'
        elif ambiguous_cr(name): role='process' if 'Cr' in elements else 'review'
        elif TARGET_RE.search(name): role='target'
        elif META_RE.search(name): role='metadata'
        elif NUM_RE.search(name) or CAT_RE.search(name): role='process'
        else: role='review'
        unique=int(s.nunique()); rec=dict(role=role,type=typ,missing=int(s.isna().sum()),unique=unique,numeric_count=int(v.notna().sum()),examples=s.dropna().drop_duplicates().head(5).tolist())
        if typ=='numeric' and v.notna().any(): rec.update(min=float(v.min()),max=float(v.max()),median=float(v.median()))
        if typ=='categorical' and unique>30: rec['warning']='High cardinality; check IDs, units or free text.'
        if typ=='categorical' and NUM_RE.search(name): rec['warning']='Non-numeric values in a parameter-like column; verify units or select numeric after cleaning.'
        if ambiguous_cr(name):
            rec['warning']='CR may mean cold-rolling reduction, not chromium; verify the original definition and units. / CR可能表示冷轧压下率，非铬含量；请核对原始定义及单位。'
        elif role=='review' and bare_uppercase_alias(name) and identify(name,elements)[0] in elements:
            rec['warning']='Ambiguous uppercase abbreviation preserved separately from the composition column; verify its role. / 大写缩写与元素列冲突，已保留原列，请核对字段角色。'
        out[name]=rec
    return out

def categorical_columns(df,c):
    extras=c.get('extras',[])
    cat=c.get('categorical_extras')
    if cat is None: return [e for e in extras if process_type(df[e],e)=='categorical']
    if not isinstance(cat,list) or len(set(cat))!=len(cat) or not set(cat)<=set(extras): raise ValueError('Categorical process fields must be a subset of selected extra inputs.')
    return cat

def process_frame(df,extras,categorical):
    out={}
    for e in extras:
        if e in categorical: out[e]=clean_text(df[e]).astype(object)
        else:
            s=clean_text(df[e]); v=numeric(s)
            if (s.notna() & v.isna()).any(): raise ValueError(f'{e}: invalid numeric process values; clean units or select categorical type.')
            out[e]=v.astype(float)
    return pd.DataFrame(out,index=df.index).reset_index(drop=True)

def process_summary(df,c):
    extras=c.get('extras',[]); cat=categorical_columns(df,c); a=process_frame(df,extras,cat); fields={}
    for e in extras:
        s=a[e].dropna()
        if not len(s): raise ValueError(f'{e}: process column has no usable values.')
        fields[e]=dict(type='categorical',values=sorted(s.unique().tolist()),missing=int(a[e].isna().sum())) if e in cat else dict(type='numeric',min=float(s.min()),max=float(s.max()),median=float(s.median()),missing=int(a[e].isna().sum()))
    complete=a.dropna().drop_duplicates() if extras else a.iloc[:1]
    # Default to an actual complete row closest to numeric medians and category modes.
    if extras and len(complete):
        score=np.zeros(len(complete))
        for e in extras:
            s=a[e].dropna()
            if e in cat: score+=(complete[e]!=s.mode().iloc[0]).to_numpy(float)
            else: score+=abs(complete[e].to_numpy()-s.median())/max(float(s.max()-s.min()),1e-12)
        default=complete.iloc[int(np.argmin(score))].to_dict()
    else: default={e:(fields[e]['values'][0] if e in cat else fields[e]['median']) for e in extras}
    return dict(fields=fields,default=default,observed_combinations=int(len(complete)) if extras else 0)


def constrained_process_pool(df, extras, categorical, mode, conditions, ranges, categories, results=None, targets=()):
    """Recorded complete tuples; bounded mode uses these as anchors for unspecified fields.
    A numeric bound is a hard filter in observed mode and a sampling domain in bounded mode.
    """
    a=process_frame(df,extras,categorical).dropna().drop_duplicates()
    for e in categorical:
        if e in categories: a=a[a[e].isin([str(v).strip() for v in categories[e]])]
        if e in conditions: a=a[a[e]==str(conditions[e]).strip()]
        if results is not None:
            for t in targets:
                if e in results[t]['cols']: a=a[a[e].isin(results[t]['train_x'][e].dropna())]
    if mode=='observed':
        for e,(lo,hi) in ranges.items(): a=a[a[e].between(lo,hi)]
        for e,v in conditions.items():
            if e not in categorical: a=a[np.isclose(a[e],float(v),rtol=0,atol=1e-9)]
    if not len(a): raise ValueError('No complete recorded process combination satisfies the requested constraints and supported categories / 无符合约束的完整工艺组合。Consider bounded mode for numerical ranges or revise constraints.')
    return a.reset_index(drop=True)
