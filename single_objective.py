"""Finite-pool single-objective recommendations and conditional envelope."""
import numpy as np
import pandas as pd

def single_results(out,target,direction,top_n,axes=None):
    y='pred_'+target
    f=out[out.Feasible].sort_values([y,'pool_index'],ascending=[direction=='min',True],kind='stable').copy()
    f['objective_rank']=np.arange(1,len(f)+1)
    top=f.head(top_n).copy(); envelope=pd.DataFrame()
    if axes and len(f):
        x=axes.get('x_element') or axes.get('x_process'); v=pd.to_numeric(f[x],errors='raise'); width=axes.get('bin_width')
        if width is None and 'x_process' in axes and v.nunique()>30:
            width=float((v.max()-v.min())/20)
        if width and v.max()>v.min():
            origin=float(v.min()); bins=np.floor((v-origin)/width+1e-10).astype(int); group=bins
            f['envelope_bin_lower']=origin+bins*width; f['envelope_bin_upper']=origin+(bins+1)*width
        else: group=v.round(10)
        f['envelope_group_count']=group.map(group.value_counts())
        envelope=f.loc[~group.duplicated()].sort_values(x).copy()
    return f,top,envelope
