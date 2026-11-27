"""Shared, conservative design-constraint parser. Never widen an explicit constraint."""
import re, math, unicodedata
from mapping import identify


def token(x):
    return re.sub(r'[\s_\-]+',' ',unicodedata.normalize('NFKC',str(x)).strip().lower())


def field_key(key, fields):
    if key in fields: return key
    matches=[k for k in fields if token(k)==token(key)]
    return matches[0] if len(matches)==1 else key


def element_key(key, elements):
    # Explicit weight percentages are not equivalent to the at.% design domain.
    e,u=identify(str(key),elements)
    if u not in (None,'at.%'): return key
    return e if e in elements else field_key(key,elements)


def number(v, atomic=False):
    if isinstance(v,bool): return v
    if isinstance(v,str):
        v=unicodedata.normalize('NFKC',v).strip()
        if atomic: v=re.sub(r'\s*(?:at\.?\s*%|原子百分比|原子%)$','',v,flags=re.I)
        try: return float(v) if math.isfinite(float(v)) else v
        except ValueError: return v
    return v


def interval(v, atomic=False):
    if isinstance(v,dict):
        for a,b in [('min','max'),('lower','upper'),('low','high')]:
            if set(v)=={a,b}: v=[v[a],v[b]]; break
        else:
            if set(v)=={'value'}: v=v['value']
    if isinstance(v,str):
        m=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)\s*(?:–|—|~|至|-)\s*([+-]?\d+(?:\.\d+)?)\s*(at\.?\s*%)?\s*',v,re.I)
        if m and (atomic or not m[3]): v=[m[1],m[2]]
    if isinstance(v,(list,tuple)): return [number(x,atomic) for x in v]
    v=number(v,atomic)
    return [v,v] if isinstance(v,(float,int)) and not isinstance(v,bool) else v


def remap(raw, keyfunc, valuefunc, label):
    if not isinstance(raw,dict): return raw
    out={}
    for k,v in raw.items():
        nk=keyfunc(k); nv=valuefunc(nk,v)
        if nk in out and out[nk]!=nv:
            # Keep conflicting alias visible so approval cannot silently choose one.
            out[k if k!=nk else '__duplicate__'+str(k)]=nv
        else: out[nk]=nv
    return out


def normalize_design(default, proposed, c, df):
    d={**default,**proposed}; notes=[]; sources={k:('LLM' if k in proposed else 'design_default') for k in d}
    els=c['elements']; extras=c['extras']; from process import categorical_columns
    cats=set(categorical_columns(df,c)); ek=lambda k:element_key(k,els); pk=lambda k:field_key(k,extras)
    if 'bounds' in proposed:
        raw=remap(proposed['bounds'],ek,lambda k,v:interval(v,True),'bounds')
        if isinstance(raw,dict):
            d['bounds']={**default['bounds'],**raw}; missing=[e for e in els if e not in raw]
            if missing:
                sources['bounds']='LLM_partial+observed_ranges'; notes.append('未指定元素使用数据集范围 / Unspecified elements retain observed at.% bounds: '+', '.join(f'{e}={d["bounds"][e]}' for e in missing)+'. No automatic widening.')
        else: d['bounds']=raw
    # Accept common range aliases, but preserve conflicts for validation.
    ranges={}; categories={}; supplied=proposed.get('conditions',{})
    def combine(dest,raw,label,func):
        if not isinstance(raw,dict): return raw
        if not isinstance(dest,dict): return dest
        for k,v in remap(raw,pk,func,label).items():
            if k in dest and dest[k]!=v: dest['__conflict__'+str(k)]=v
            else: dest[k]=v
        return dest
    for name in ('process_bounds','processing_bounds','process_ranges'):
        if name in proposed: ranges=combine(ranges,proposed[name],name,lambda k,v:interval(v))
        if name!='process_ranges': d.pop(name,None)
    if 'process_categories' in proposed: categories=combine(categories,proposed['process_categories'],'categories',lambda k,v:list(v) if isinstance(v,(tuple,list)) else [v])
    conditions={}
    if isinstance(supplied,dict):
        for k,v in remap(supplied,pk,lambda k,v:v,'conditions').items():
            if k in cats and isinstance(v,(list,tuple)):
                categories=combine(categories,{k:v},'categories',lambda k,v:list(v))
            elif k not in cats and isinstance(v,(list,tuple,dict)):
                ranges=combine(ranges,{k:v},'ranges',lambda k,v:interval(v))
            elif k not in cats and isinstance(v,str) and isinstance(interval(v),list) and interval(v)[0]!=interval(v)[1]:
                ranges=combine(ranges,{k:v},'ranges',lambda k,v:interval(v))
            else: conditions[k]=v if k in cats else number(v)
    else: conditions=supplied
    mode=d.get('process_mode','fixed'); t=token(mode)
    aliases={'fixed':'fixed','fixed process':'fixed','fixed processing':'fixed','fixed conditions':'fixed','固定工艺':'fixed','固定工艺条件':'fixed','observed':'observed','observed process':'observed','observed combinations':'observed','observed process combinations':'observed','joint observed':'observed','已有工艺组合':'observed','联合搜索已记录工艺组合':'observed','composition only':'fixed','none':'fixed','no process':'fixed','仅成分':'fixed','无工艺':'fixed','bounded':'bounded','bounded process':'bounded','continuous':'bounded','continuous process':'bounded','continuous optimization':'bounded','range':'bounded','joint bounded':'bounded','工艺范围寻优':'bounded','连续工艺寻优':'bounded','joint':'observed','joint optimization':'observed','joint search':'observed','联合寻优':'observed'}
    mapped=aliases.get(t,mode)
    if (ranges or categories) and ('process_mode' not in proposed or t in ('joint','joint optimization','joint search','联合寻优')):
        mapped='bounded' if ranges else 'observed'
        notes.append('Process constraints supplied; proposed range search mode for review / 已提供工艺约束，建议范围寻优模式，请审核。')
    if mapped!=mode: notes.append(f'Process mode {mode!r} normalized to {mapped!r}.'); sources['process_mode']='LLM_normalized'
    if mapped in ('observed','bounded') and not extras and not (ranges or categories or conditions):
        mapped='fixed'; notes.append('No process inputs; composition-only design.'); sources['process_mode']='no_process_inputs'
    if mapped=='fixed' and isinstance(conditions,dict) and not (ranges or categories):
        missing=set(extras)-set(conditions); conditions={**default.get('conditions',{}),**conditions}
        if missing: notes.append('Unspecified fixed process inputs use representative recorded values / 未指定固定工艺采用代表性记录值: '+', '.join(sorted(missing))); sources['conditions']='LLM_partial+observed_default' if 'conditions' in proposed else 'observed_default'
    if mapped=='bounded': notes.append('Bounded mode samples specified numerical ranges and may produce unseen process combinations; verify physical feasibility / 范围寻优可能产生未见过的工艺组合，需核实可制造性。')
    d.update(process_mode=mapped,conditions=conditions,process_ranges=ranges,process_categories=categories)
    if 'directions' in d and isinstance(d['directions'],dict):
        d['directions']=remap(d['directions'],lambda k:field_key(k,c['targets']),lambda k,v:{'maximize':'max','minimize':'min','maximum':'max','minimum':'min','最大化':'max','最小化':'min'}.get(token(v),v),'directions')
    if 'thresholds' in d: d['thresholds']=remap(d['thresholds'],lambda k:field_key(k,c['targets']),lambda k,v:number(v),'thresholds')
    for k in ('step','pool_size','iterations','min_entropy','top_n'):
        if k in d: d[k]=number(d[k])
    if 'step' not in proposed and isinstance(d.get('bounds'),dict):
        # Refine an omitted step only, never an explicitly requested step.
        from engine import pool_grid
        for step in (d['step'],.5,.25,.2,.1):
            try: pool_grid(els,d['bounds'],step,1,42)
            except (ValueError,TypeError,KeyError,OverflowError): continue
            if step!=d['step']: d['step']=step; sources['step']='grid_feasibility'; notes.append(f'Unspecified step refined to {step:g} at.% to respect bounds.')
            break
    if isinstance(d.get('plot_axes'),dict):
        axes=dict(d['plot_axes'])
        if 'x_element' in axes: axes['x_element']=ek(axes['x_element'])
        if 'x_process' in axes: axes['x_process']=pk(axes['x_process'])
        if 'y_target' in axes: axes['y_target']=field_key(axes['y_target'],c['targets'])
        if 'bin_width' in axes: axes['bin_width']=number(axes['bin_width'])
        d['plot_axes']=axes
    return d,notes,sources
