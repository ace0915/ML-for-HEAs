"""Conservative composition-header normalization and provisional unit detection."""
import re, unicodedata
import numpy as np
import pandas as pd

CHINESE=dict(zip('铁 镍 钴 铬 铝 铜 锰 钼 钛 钒 钽 钨 铌 锆 铪 硅 碳 硼 镁 锌 锡 银 金 铅 磷 硫 氮 氧'.split(),'Fe Ni Co Cr Al Cu Mn Mo Ti V Ta W Nb Zr Hf Si C B Mg Zn Sn Ag Au Pb P S N O'.split()))
ENGLISH=dict(zip('iron nickel cobalt chromium aluminum aluminium copper manganese molybdenum titanium vanadium tantalum tungsten niobium zirconium hafnium silicon carbon boron magnesium zinc tin silver gold lead phosphorus sulfur nitrogen oxygen'.split(),'Fe Ni Co Cr Al Al Cu Mn Mo Ti V Ta W Nb Zr Hf Si C B Mg Zn Sn Ag Au Pb P S N O'.split()))
AT=r'(?:at\.?\s*%|atomic\s*(?:percent(?:age)?|%)|原子百分比|原子\s*%)'
WT=r'(?:wt\.?\s*%|weight\s*(?:percent(?:age)?|%)|mass\s*(?:percent(?:age)?|%)|质量百分比|质量\s*%)'
FR=r'(?:atomic\s*fraction|mole\s*fraction|原子分数|摩尔分数)'
UNIT=re.compile(f'{AT}|{WT}|{FR}',re.I)

def bare_uppercase_alias(name):
    """An all-caps two-letter abbreviation without an explicit composition marker."""
    n=unicodedata.normalize('NFKC',str(name)).strip()
    return re.fullmatch(r'[A-Z]{2}(?:[ _]*\(?\s*%\s*\)?)?',n) is not None

def ambiguous_cr(name):
    n=unicodedata.normalize('NFKC',str(name)).strip()
    return bare_uppercase_alias(n) and n.startswith('CR')

def identify(name,elements):
    n=unicodedata.normalize('NFKC',name).strip()
    if n in elements: return n,None
    # CR or CR(%) is commonly a processing abbreviation, not a composition label.
    # Explicit CR(at%) / CR(wt%) and correctly cased Cr remain valid element headers.
    if ambiguous_cr(n): return None,None
    units=set('at.%' if re.fullmatch(AT,m.group(),re.I) else 'wt.%' if re.fullmatch(WT,m.group(),re.I) else 'fraction' for m in UNIT.finditer(n))
    if len(units)>1: return None,None
    s=UNIT.sub('',n)
    s=re.sub(r'(?:含量|元素|content|composition|concentration)', '',s,flags=re.I)
    s=re.sub(r'[\s_()\[\]{}%:.-]+','',s)
    # Bare unknown case variants of one-letter symbols can be process codes; require exact case or explicit composition marker.
    explicit=s!=n and (bool(units) or bool(re.search(r'含量|元素|content|composition|concentration|%',n,re.I)))
    sym=s if s in elements else CHINESE.get(s) or ENGLISH.get(s.lower())
    if sym is None and (len(s)>1 or explicit): sym=next((e for e in elements if e.lower()==s.lower()),None)
    return sym, next(iter(units),None)

def normalize_headers(df,elements):
    matches={}; hints={}; groups={}
    for col in df.columns:
        sym,unit=identify(col,elements)
        if sym and pd.to_numeric(df[col],errors='coerce').notna().any():
            groups.setdefault(sym,[]).append(col); hints[col]=unit
    # If a clearly labelled element has a competing weak uppercase abbreviation,
    # keep the weak field untouched for human role review, not as a second element.
    for e,cols in list(groups.items()):
        if len(cols)>1:
            strong=[col for col in cols if not bare_uppercase_alias(col)]
            if len(strong)==1:
                groups[e]=strong
    conflicts={e:cols for e,cols in groups.items() if len(cols)>1}
    if conflicts:
        raise ValueError('Ambiguous composition columns / 元素列重复: '+str(conflicts)+'; keep one column per element.')
    for sym,cols in groups.items(): matches[cols[0]]=sym
    units={sym:hints[col] for col,sym in matches.items() if hints[col]}
    return df.rename(columns=matches),matches,units

def proposed_unit(df,els,hints):
    units={hints[e] for e in els if e in hints}
    if len(units)>1: raise ValueError('Mixed composition units / 元素列单位混用，请先统一为同一单位。')
    if units: return next(iter(units)),False
    if not els: return None,True
    x=df[els].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not np.isfinite(x).all() or (x<0).any(): return None,True
    totals=x.sum(1)
    if np.all(abs(totals-1)<=.02): return 'fraction',True
    if np.all(abs(totals-100)<=2): return 'at.%',True
    return None,True
