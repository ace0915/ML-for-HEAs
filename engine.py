"""Deterministic numerical engine. No LLM-generated code execution."""
import numpy as np, pandas as pd, warnings, math
from functools import lru_cache
from scipy.stats import spearmanr, rankdata
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from process import process_frame, process_type, constrained_process_pool
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split, cross_val_score
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge,HuberRegressor,ARDRegression,SGDRegressor,OrthogonalMatchingPursuit,PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor,RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.svm import SVR,NuSVR,LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore',category=ConvergenceWarning)
ELEMENTS='H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()
DESCRIPTORS=['S_mix_J_molK','N_elements','c_max','c_min_positive','r_mean_pm','delta_r_pct','r_range_pm','VEC_mean','VEC_std','chi_mean','chi_std','chi_range','Tm_mean_K','Tm_std_K','Tm_range_K','H_mix_kJ_mol','Omega','Lambda_S_over_deltaPct2','G_mix_at_Tm_kJ_mol','mass_mean','mass_std','Z_mean','Z_std','Mendeleev_mean','Mendeleev_std','s_valence_mean','p_valence_mean','d_valence_mean','f_valence_mean','GSvolume_mean_A3']
SELECTORS=['Pearson','Spearman','Mutual information','F test','Variance','Random forest importance','Extra trees importance','Lasso','RFE (Ridge)','Sequential forward (Ridge)']
OPTIMIZERS=['None','Genetic algorithm','Simulated annealing','Particle swarm','Ant colony']
@lru_cache(None)
def sources():
    from matminer.utils.data import MagpieData,MixingEnthalpy
    from pymatgen.core import Element
    return MagpieData(impute_nan=False),MixingEnthalpy(impute_nan=False),Element
@lru_cache(None)
def prop(e,p):
    try:
        m,_,E=sources(); return float(m.get_elemental_property(E(e),p))
    except (KeyError,ValueError,TypeError,IndexError): return np.nan
@lru_cache(None)
def pair(a,b):
    try:
        _,m,E=sources(); return float(m.get_mixing_enthalpy(E(a),E(b)))
    except (KeyError,ValueError,TypeError,IndexError): return np.nan

def fractions(df,els,unit='at.%',tol=2):
    x=df[els].apply(pd.to_numeric,errors='raise').to_numpy(float)
    total=x.sum(1); expected=1 if unit=='fraction' else 100
    if not np.isfinite(x).all() or (x<0).any() or (total<=0).any(): raise ValueError('成分存在缺失、负数或零总量。')
    if (abs(total-expected)>tol*expected/100).any(): raise ValueError(f'成分合计偏离 {expected} 超过 {tol}%；请核查元素列及单位。')
    if unit=='wt.%':
        weights=np.array([prop(e,'AtomicWeight') for e in els]);
        if not np.isfinite(weights).all(): raise ValueError('原子量缺失，不能转换。')
        x=x/weights
    return x/x.sum(1,keepdims=True)

def descriptors(x,els,requested=DESCRIPTORS):
    x=np.asarray(x,float); out={}; pos=x>0
    with np.errstate(divide='ignore',invalid='ignore'):
        S=-8.314462618*np.sum(np.where(pos,x*np.log(x),0),axis=1)
    out.update(S_mix_J_molK=S,N_elements=pos.sum(1),c_max=x.max(1),c_min_positive=np.where(pos,x,np.inf).min(1))
    need=set(requested)-set(out)
    if need:
        def stats(p):
            a=np.array([prop(e,p) for e in els]); valid=~np.any(pos & ~np.isfinite(a)[None,:],axis=1); a=np.nan_to_num(a,nan=0.0)
            mean=x@a; std=np.sqrt(np.sum(x*(a-mean[:,None])**2,axis=1)); rg=np.where(pos,a,-np.inf).max(1)-np.where(pos,a,np.inf).min(1)
            return tuple(np.where(valid,z,np.nan) for z in [mean,std,rg])
        for p,names in [('MiracleRadius',['r_mean_pm','delta_r_pct','r_range_pm']),('NValence',['VEC_mean','VEC_std']),('Electronegativity',['chi_mean','chi_std','chi_range']),('MeltingT',['Tm_mean_K','Tm_std_K','Tm_range_K']),('AtomicWeight',['mass_mean','mass_std']),('Number',['Z_mean','Z_std']),('MendeleevNumber',['Mendeleev_mean','Mendeleev_std']),('NsValence',['s_valence_mean']),('NpValence',['p_valence_mean']),('NdValence',['d_valence_mean']),('NfValence',['f_valence_mean']),('GSvolume_pa',['GSvolume_mean_A3'])]:
            extra={'MiracleRadius':{'Lambda_S_over_deltaPct2'},'MeltingT':{'Omega','G_mix_at_Tm_kJ_mol'}}.get(p,set())
            if not (set(names)|extra)&need: continue
            a,b,c=stats(p)
            if p=='MiracleRadius': b=100*b/np.where(a!=0,a,np.nan)
            out.update(zip(names,[a,b,c]))
        if need & {'H_mix_kJ_mol','Omega','G_mix_at_Tm_kJ_mol'}:
            h=np.zeros(len(x)); valid=np.ones(len(x),bool)
            for i,a in enumerate(els):
                for j,b in enumerate(els[i+1:],i+1):
                    if not np.any(pos[:,i]&pos[:,j]): continue
                    v=pair(*sorted([a,b]));
                    if not np.isfinite(v): valid &= ~(pos[:,i]&pos[:,j]); continue
                    h+=4*x[:,i]*x[:,j]*v
            h=np.where(valid,h,np.nan); out['H_mix_kJ_mol']=h
            if 'Omega' in need: out['Omega']=out['Tm_mean_K']*S/1000/np.where(abs(h)>1e-12,abs(h),np.nan)
            if 'G_mix_at_Tm_kJ_mol' in need: out['G_mix_at_Tm_kJ_mol']=h-out['Tm_mean_K']*S/1000
        if 'Lambda_S_over_deltaPct2' in need: out['Lambda_S_over_deltaPct2']=S/np.where(out['delta_r_pct']>0,out['delta_r_pct']**2,np.nan) # empirical S/delta_percent^2
    return pd.DataFrame({k:out[k] for k in requested})
def features(df,els,unit,desc,extras,raw=True,categorical=None):
    x=fractions(df,els,unit); a=pd.DataFrame(x*100,columns=els) if raw else pd.DataFrame(index=range(len(df)))
    b=descriptors(x,els,desc); categorical=[e for e in extras if process_type(df[e],e)=='categorical'] if categorical is None else categorical; c=process_frame(df,extras,categorical)
    return pd.concat([a,b,c],axis=1).replace([np.inf,-np.inf],np.nan)

class SelectFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,methods=('Pearson',),k=15,seed=42): self.methods=methods; self.k=k; self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X); y=np.asarray(y).ravel(); p=X.shape[1]
        if not self.methods or not set(self.methods)<=set(SELECTORS): raise ValueError('Feature evaluation and selection are mandatory; choose at least one registered method.')
        if p<1 or int(self.k)!=self.k or self.k<1: raise ValueError('Top-K must be a positive integer and at least one input must be available.')
        k=min(int(self.k),p); ranks=[]; self.method_scores_={}; self.requested_k_=int(self.k); self.effective_k_=k
        if len(y)<3 or not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('Feature evaluation needs finite preprocessed inputs and at least three training rows.')
        for m in self.methods:
            if m=='Pearson': s=[abs(np.corrcoef(X[:,i],y)[0,1]) if np.std(X[:,i]) and np.std(y) else 0 for i in range(p)]
            elif m=='Spearman': s=[abs(spearmanr(X[:,i],y).statistic) if np.std(X[:,i]) and np.std(y) else 0 for i in range(p)]
            elif m=='Mutual information': s=mutual_info_regression(X,y,n_neighbors=min(3,len(y)-1),random_state=self.seed)
            elif m=='F test': s=f_regression(X,y)[0]
            elif m=='Variance': s=np.var(X,axis=0)
            elif m=='Random forest importance': s=RandomForestRegressor(n_estimators=60,random_state=self.seed,n_jobs=1).fit(X,y).feature_importances_
            elif m=='Extra trees importance': s=ExtraTreesRegressor(n_estimators=60,random_state=self.seed,n_jobs=1).fit(X,y).feature_importances_
            elif m=='Lasso': s=abs(Lasso(alpha=max(np.std(y)*.01,1e-6),max_iter=4000).fit(StandardScaler().fit_transform(X),y).coef_)
            elif m=='RFE (Ridge)': s=np.array([1.0]) if p==1 else -RFE(Ridge(),n_features_to_select=1,step=.2).fit(StandardScaler().fit_transform(X),y).ranking_
            elif m=='Sequential forward (Ridge)':
                if p==1: s=np.array([1.0]) # only one possible feature
                else: s=SequentialFeatureSelector(Pipeline([('scale',StandardScaler()),('ridge',Ridge())]),n_features_to_select=min(k,p-1),cv=min(3,len(y)),scoring='neg_mean_squared_error').fit(X,y).get_support().astype(float)
            else: raise ValueError(m)
            s=np.asarray(s,dtype=float)
            if s.shape!=(p,) or not np.isfinite(s).all(): raise ValueError(f'{m}: feature evaluation produced invalid scores; inspect the data or choose another method.')
            normalized=rankdata(s,method='average')/p
            self.method_scores_[m]=normalized
            ranks.append(normalized)
        self.scores_=np.mean(ranks,axis=0)
        self.indices_=np.sort(np.argsort(-self.scores_,kind='stable')[:k])
        return self
    def transform(self,X): return np.asarray(X)[:,self.indices_]

def model_specs(seed=42,n=100):
    # constructors and finite discrete hyperparameter domains
    return {
    'Linear regression':(LinearRegression(),{'fit_intercept':[True,False]}),
    'Ridge':(Ridge(),{'alpha':[.01,.1,1,10,100]}),
    'Lasso':(Lasso(max_iter=6000),{'alpha':[.001,.01,.1,1,10]}),
    'ElasticNet':(ElasticNet(max_iter=6000),{'alpha':[.001,.01,.1,1,10],'l1_ratio':[.1,.5,.9]}),
    'Bayesian ridge':(BayesianRidge(),{'alpha_1':[1e-7,1e-5,.001],'lambda_1':[1e-7,1e-5,.001]}),
    'ARD regression':(ARDRegression(),{'threshold_lambda':[100,1000,10000]}),
    'Huber':(HuberRegressor(max_iter=1000),{'epsilon':[1.2,1.35,1.8],'alpha':[.0001,.01,1]}),
    'SGD':(SGDRegressor(random_state=seed,max_iter=3000),{'alpha':[.0001,.001,.01],'penalty':['l2','l1','elasticnet']}),
    'OMP':(OrthogonalMatchingPursuit(),{'tol':[.01,1,100]}),
    'PLS':(PLSRegression(n_components=1),{'scale':[True,False]}),
    'Decision tree':(DecisionTreeRegressor(random_state=seed),{'max_depth':[3,6,12,None],'min_samples_leaf':[1,2,4]}),
    'Extra tree':(ExtraTreeRegressor(random_state=seed),{'max_depth':[3,6,12,None],'min_samples_leaf':[1,2,4]}),
    'Random forest':(RandomForestRegressor(n_estimators=100,n_jobs=1,random_state=seed),{'n_estimators':[60,120,240],'max_depth':[5,12,None],'min_samples_leaf':[1,2,4]}),
    'Extra trees':(ExtraTreesRegressor(n_estimators=100,n_jobs=1,random_state=seed),{'n_estimators':[60,120,240],'max_features':[.5,.8,1.],'min_samples_leaf':[1,2,4]}),
    'Gradient boosting':(GradientBoostingRegressor(random_state=seed),{'n_estimators':[60,120,240],'learning_rate':[.03,.1,.2],'max_depth':[2,3,5]}),
    'Hist gradient boosting':(HistGradientBoostingRegressor(random_state=seed),{'max_iter':[60,120,240],'learning_rate':[.03,.1,.2],'max_leaf_nodes':[7,15,31]}),
    'AdaBoost':(AdaBoostRegressor(random_state=seed),{'n_estimators':[30,80,150],'learning_rate':[.03,.1,1.]}),
    'Bagging':(BaggingRegressor(random_state=seed,n_jobs=1),{'n_estimators':[20,50,100],'max_samples':[.6,.8,1.]}),
    'SVR RBF':(SVR(),{'C':[1,10,100,1000],'gamma':['scale',.01,.1,1],'epsilon':[.01,.1,1]}),
    'SVR polynomial':(SVR(kernel='poly'),{'C':[1,10,100],'degree':[2,3,4],'epsilon':[.1,1]}),
    'NuSVR':(NuSVR(),{'C':[1,10,100,1000],'nu':[.25,.5,.75]}),
    'Linear SVR':(LinearSVR(random_state=seed,max_iter=6000),{'C':[.1,1,10,100],'epsilon':[0,.1,1]}),
    'KNN':(KNeighborsRegressor(n_neighbors=3),{'n_neighbors':[1,2,3,5],'weights':['uniform','distance'],'p':[1,2]}),
    'Kernel ridge':(KernelRidge(kernel='rbf'),{'alpha':[.001,.01,.1,1],'gamma':[.01,.1,1]}),
    'Gaussian process':(GaussianProcessRegressor(kernel=Matern(nu=2.5)+WhiteKernel(.01),normalize_y=True,random_state=seed),{'alpha':[1e-8,1e-5,.001]}),
    'MLP':(MLPRegressor(max_iter=1000,random_state=seed),{'hidden_layer_sizes':[(32,),(64,),(64,32)],'alpha':[.0001,.01,1],'learning_rate_init':[.001,.01]}),
    'XGBoost':(None,{'n_estimators':[60,120,240],'max_depth':[2,4,6],'learning_rate':[.03,.1,.2]}),
    'LightGBM':(None,{'n_estimators':[60,120,240],'num_leaves':[7,15,31],'learning_rate':[.03,.1,.2]}),
    'CatBoost':(None,{'iterations':[60,120,240],'depth':[3,5,7],'learning_rate':[.03,.1,.2]}),
    'Polynomial ridge':(Pipeline([('poly',PolynomialFeatures(include_bias=False)),('ridge',Ridge())]),{'poly__degree':[2,3],'ridge__alpha':[.1,1,10,100]})}

class CatBoostAdapter(RegressorMixin, BaseEstimator):
    def __init__(self,iterations=100,depth=6,learning_rate=.1,seed=42):
        self.iterations=iterations; self.depth=depth; self.learning_rate=learning_rate; self.seed=seed
    def fit(self,X,y):
        from catboost import CatBoostRegressor
        self.model_=CatBoostRegressor(iterations=self.iterations,depth=self.depth,learning_rate=self.learning_rate,random_seed=self.seed,thread_count=1,verbose=False,allow_writing_files=False).fit(X,y)
        self.n_features_in_=np.asarray(X).shape[1]; return self
    def predict(self,X): return self.model_.predict(X)

def get_model(name,seed):
    model,space=model_specs(seed)[name]
    if name=='XGBoost':
        from xgboost import XGBRegressor
        model=XGBRegressor(random_state=seed,n_jobs=1,verbosity=0,n_estimators=100)
    elif name=='LightGBM':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(random_state=seed,n_jobs=1,verbosity=-1,n_estimators=100)
    elif name=='CatBoost':
        model=CatBoostAdapter(seed=seed)
    return model,space

def search(space,score,method,budget,seed):
    rng=np.random.default_rng(seed); keys=list(space); sizes=np.array([len(space[k]) for k in keys]); cache={}; trace=[]
    def evaluate(x):
        x=tuple(np.clip(np.rint(x),0,sizes-1).astype(int)); params={k:space[k][i] for k,i in zip(keys,x)}
        if x not in cache:
            try: val=float(score(params)); val=val if np.isfinite(val) else np.inf
            except Exception: val=np.inf
            cache[x]=val; trace.append({'trial':len(trace)+1,'cv_rmse':val,'params':params})
        return cache[x]
    if method=='None': return {},trace
    def rand(): return np.array([rng.integers(s) for s in sizes])
    pop=np.array([rand() for _ in range(min(6,budget))]); velocity=np.zeros_like(pop,dtype=float); personal=pop.copy(); pf=np.full(len(pop),np.inf); pher=[np.ones(s) for s in sizes]; current=rand(); curval=evaluate(current)
    for t in range(max(1,budget*5)):
        if len(cache)>=min(budget,int(np.prod(sizes))): break
        if method=='Simulated annealing':
            a=current.copy(); j=rng.integers(len(keys)); a[j]=rng.integers(sizes[j]); v=evaluate(a); temp=max(1e-6,(abs(curval) if np.isfinite(curval) else 1)*.2*(.94**t));
            if v<curval or (np.isfinite(v) and rng.random()<np.exp(np.clip((curval-v)/temp,-700,0))): current,curval=a,v
        elif method=='Ant colony':
            a=np.array([rng.choice(s,p=p/p.sum()) for s,p in zip(sizes,pher)]); v=evaluate(a)
            for p in pher: p*=.9; p+=.05
            if np.isfinite(v) and v<=min(cache.values()):
                for p,j in zip(pher,a): p[j]+=1
        elif method=='Particle swarm':
            i=t%len(pop); a=pop[i]; v=evaluate(a)
            if v<pf[i]: personal[i]=a; pf[i]=v
            best=np.array(min(cache,key=cache.get)); velocity[i]=.6*velocity[i]+1.4*rng.random(len(keys))*(personal[i]-a)+1.4*rng.random(len(keys))*(best-a); pop[i]=np.clip(np.rint(a+velocity[i]),0,sizes-1).astype(int)
            if rng.random()<.25: pop[i]=rand()
        else: # elitist discrete genetic search
            a=pop[t%len(pop)].copy() if t<len(pop) else None
            if a is None:
                elite=sorted(cache,key=cache.get)[:max(1,min(4,len(cache)))]; p,q=elite[rng.integers(len(elite))],elite[rng.integers(len(elite))]; a=np.where(rng.random(len(keys))<.5,p,q); j=rng.integers(len(keys)); a[j]=rng.integers(sizes[j])
            evaluate(a)
        if t%3==0 and len(cache)<min(budget,int(np.prod(sizes))): evaluate(rand())
    if not cache or not np.isfinite(min(cache.values())): raise ValueError('所有超参数试验均失败，请检查样本和模型。')
    best=min(cache,key=cache.get); return {k:space[k][i] for k,i in zip(keys,best)},trace

def adaptive_space(name,space,n,y_train,p):
    """Adjust selected discrete search domains using outer-training data only."""
    space={k:list(v) for k,v in space.items()}; spread=max(float(np.std(y_train)),1e-8)
    if name in ('SVR RBF','SVR polynomial','NuSVR','Linear SVR'):
        if 'C' in space: space['C']=[float(spread*v) for v in (.1,1,10,100)]
        if 'epsilon' in space: space['epsilon']=[float(spread*v) for v in (.005,.02,.1,.2)]
    if 'max_depth' in space:
        upper=max(2,min(16,int(np.log2(max(n,4)))))
        finite=[v for v in space['max_depth'] if v is not None and v<=upper]
        space['max_depth']=sorted(set(finite+[max(2,upper//2),upper]))+([None] if None in space['max_depth'] and n>=100 else [])
    if 'min_samples_leaf' in space:
        space['min_samples_leaf']=sorted(set([1,2,max(2,min(16,n//30)),max(2,min(32,n//15))]))
    if name=='Ridge':
        space['alpha']=[.01,.1,1,10,100] if n>=5*p else [.1,1,10,100,1000]
    if name=='KNN': space['n_neighbors']=[v for v in (1,2,3,5,9,15) if v<=n]
    return space

def metrics(y,p):
    y=np.asarray(y).ravel(); p=np.asarray(p).ravel(); nz=abs(y)>1e-12
    return {'R2':float(r2_score(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),'MAPE_pct':float(np.mean(abs((y[nz]-p[nz])/y[nz]))*100) if nz.any() else np.nan}

def split_target(df,target,group,test,seed,folds):
    y=pd.to_numeric(df[target],errors='coerce'); idx=np.flatnonzero(np.isfinite(y)); yy=y.iloc[idx].to_numpy(float)
    if len(yy)<12: raise ValueError(f'{target}: 有效样本不足12条。')
    if group:
        gg=df[group].iloc[idx].astype(str).to_numpy(); tr,te=next(GroupShuffleSplit(n_splits=1,test_size=test,random_state=seed).split(idx,yy,gg)); nfold=min(folds,len(set(gg[tr]))); cv=list(GroupKFold(nfold).split(tr,yy[tr],gg[tr])) if nfold>=2 else None
    else:
        tr,te=train_test_split(np.arange(len(yy)),test_size=test,random_state=seed); cv=list(KFold(min(folds,len(tr)),shuffle=True,random_state=seed).split(tr))
    if cv is None or len(te)<2 or min(len(a) for a,b in cv)<3: raise ValueError(f'{target}: 划分后样本/分组太少。')
    return idx,tr,te,cv

def feature_preprocessor(xx):
    numeric_cols=xx.select_dtypes(include='number').columns.tolist(); cat_cols=[a for a in xx.columns if a not in numeric_cols]
    return ColumnTransformer([('num',SimpleImputer(strategy='median',keep_empty_features=True),numeric_cols),('cat',Pipeline([('missing',SimpleImputer(strategy='constant',fill_value='__MISSING_PROCESS__',keep_empty_features=True)),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat_cols)],verbose_feature_names_out=True)

def feature_ranking(pre,sel):
    names=pre.get_feature_names_out().tolist()
    table=pd.DataFrame({'feature':names,'score':sel.scores_,'selected':np.isin(np.arange(len(names)),sel.indices_)})
    table['display_feature']=[name.split('__',1)[-1] for name in names]
    table['rank']=rankdata(-sel.scores_,method='average')
    for method,score in sel.method_scores_.items(): table['rank_score: '+method]=score
    return table.sort_values('score',ascending=False,kind='stable').reset_index(drop=True)

def preview_features(df,X,c):
    out={}
    for target in c['targets']:
        settings=(c.get('target_settings') or {}).get(target,c)
        idx,tr,te,cv=split_target(df,target,c.get('group'),float(c['test']),int(c['seed']),int(settings['folds']))
        xx=X.iloc[idx].reset_index(drop=True); yy=pd.to_numeric(df[target],errors='coerce').iloc[idx].to_numpy(float)
        xx=xx.loc[:,xx.iloc[tr].notna().any()]; train_x=xx.iloc[tr]
        if not len(xx.columns): raise ValueError(f'{target}: no usable training features')
        pre=feature_preprocessor(xx); encoded=pre.fit_transform(train_x)
        sel=SelectFeatures(tuple(settings['selectors']),int(settings['k']),int(c['seed'])).fit(encoded,yy[tr])
        out[target]=dict(ranking=feature_ranking(pre,sel),methods=list(sel.methods),requested_k=sel.requested_k_,effective_k=sel.effective_k_,available=encoded.shape[1],n_train=len(tr),tied=bool(np.allclose(sel.scores_,sel.scores_[0])),train_rows=idx[tr])
    return out

def train(df,X,targets,models,methods,k,optimizer,budget,folds,test,seed,group=None,progress=lambda s:None,target_settings=None):
    results={}; all_rows=[]; failures=[]; traces=[]
    common=dict(models=models,selectors=methods,k=k,optimizer=optimizer,budget=budget,folds=folds,screen_top=len(models))
    for target in targets:
        settings=(target_settings or {}).get(target,common); models=settings['models']; methods=settings['selectors']; k=settings['k']; optimizer=settings['optimizer']; budget=settings['budget']; folds=settings['folds']; screen_top=settings['screen_top']
        idx,tr,te,cv=split_target(df,target,group,test,seed,folds); yy=pd.to_numeric(df[target],errors='coerce').iloc[idx].to_numpy(float); xx=X.iloc[idx].reset_index(drop=True)
        progress(f'{target} / actual settings: {settings}')
        cols=xx.columns[xx.iloc[tr].notna().any()].tolist(); xx=xx[cols]
        if not cols: raise ValueError('训练集无有效特征。')
        candidates=[]; screened=[]
        # Hyperparameter search domains use the actual retained encoded dimension.
        retained_dim=min(k,len(feature_preprocessor(xx).fit(xx.iloc[tr]).get_feature_names_out())) if target_settings else len(cols)
        for name in models:
            progress(f'{target} / {name}')
            try:
                m,space=get_model(name,seed)
                if target_settings: space=adaptive_space(name,space,min(len(a) for a,b in cv),yy[tr],retained_dim)
                if 'n_neighbors' in space: space['n_neighbors']=[v for v in space['n_neighbors'] if v<=min(len(a) for a,b in cv)]
                pre=feature_preprocessor(xx)
                pipe=Pipeline([('preprocess',pre),('select',SelectFeatures(tuple(methods),k,seed)),('scale',StandardScaler()),('model',m)])
                def score(params):
                    q=clone(pipe).set_params(**{'model__'+a:b for a,b in params.items()}); return float(np.sqrt(-cross_val_score(q,xx.iloc[tr],yy[tr],cv=cv,scoring='neg_mean_squared_error',n_jobs=1,error_score='raise').mean()))
                if target_settings:
                    cv_rmse=score({}); screened.append((cv_rmse,name,pipe,space)); all_rows.append(dict(target=target,model=name,cv_rmse=cv_rmse,selected=False,params={},stage='screen',search_space=space))
                else:
                    params,trace=search(space,score,optimizer,budget,seed); cv_rmse=score(params); pipe.set_params(**{'model__'+a:b for a,b in params.items()}); pipe.fit(xx.iloc[tr],yy[tr]); candidates.append((cv_rmse,name,pipe,params)); traces.extend([dict(target=target,model=name,**t) for t in trace]);
            except Exception as e: failures.append({'target':target,'model':name,'error':str(e)[:500]})
        if target_settings:
            for base_rmse,name,pipe,space in sorted(screened,key=lambda z:z[0])[:screen_top]:
                progress(f'{target} / refine {name} / {optimizer} / budget {budget}')
                def score(params):
                    q=clone(pipe).set_params(**{'model__'+a:b for a,b in params.items()}); return float(np.sqrt(-cross_val_score(q,xx.iloc[tr],yy[tr],cv=cv,scoring='neg_mean_squared_error',n_jobs=1,error_score='raise').mean()))
                try:
                    params={}; cv_rmse=base_rmse
                    if optimizer!='None':
                        try:
                            tuned,trace=search(space,score,optimizer,budget,seed); tuned_rmse=score(tuned); traces.extend([dict(target=target,model=name,**t) for t in trace])
                            if tuned_rmse<cv_rmse: params=tuned; cv_rmse=tuned_rmse
                        except Exception as e: failures.append({'target':target,'model':name,'error':'Tuning failed; kept screened parameters: '+str(e)[:350]})
                    pipe.set_params(**{'model__'+a:b for a,b in params.items()}); pipe.fit(xx.iloc[tr],yy[tr]); candidates.append((cv_rmse,name,pipe,params))
                except Exception as e: failures.append({'target':target,'model':name,'error':str(e)[:500]})
        if not candidates: raise ValueError(f'{target} 所有模型失败: {failures[-3:]}')
        # Champion is selected without looking at holdout scores.
        candidates.sort(key=lambda t:t[0]); cv_rmse,name,pipe,params=candidates[0]
        for cvv,nm,pp,pa in candidates:
            all_rows.append(dict(target=target,model=nm,cv_rmse=cvv,selected=nm==name,params=pa,stage="final"))
        pred=pipe.predict(xx.iloc[te]).ravel(); met=metrics(yy[te],pred); sel=pipe.named_steps['select']; encoded_cols=pipe.named_steps['preprocess'].get_feature_names_out().tolist(); ranks=feature_ranking(pipe.named_steps['preprocess'],sel); progress(f'{target} / mandatory feature ranking: {list(sel.methods)}; requested K={sel.requested_k_}, retained={sel.effective_k_}/{len(encoded_cols)}; tied={bool(np.allclose(sel.scores_,sel.scores_[0]))}')
        results[target]=dict(model=pipe,name=name,cols=cols,metrics=met,cv_rmse=cv_rmse,params=params,ranking=ranks,encoded_cols=encoded_cols,holdout=pd.DataFrame({'row':idx[te],'actual':yy[te],'predicted':pred}),train_rows=idx[tr],test_rows=idx[te],train_x=xx.iloc[tr].copy())
    return results,pd.DataFrame(all_rows),pd.DataFrame(failures),pd.DataFrame(traces)

def pareto(values):
    a=np.asarray(values); keep=np.ones(len(a),bool)
    for i,v in enumerate(a): keep[i]=not np.any(np.all(a>=v,axis=1)&np.any(a>v,axis=1))
    return keep

def pool_grid(els,bounds,step,n,seed):
    rng=np.random.default_rng(seed); scale=round(100/step)
    if not np.isclose(scale*step,100,rtol=0,atol=1e-8): raise ValueError('Composition step must divide 100 exactly / 步长必须整除100。')
    lo=np.array([math.ceil(bounds[e][0]/step-1e-8) for e in els]); hi=np.array([math.floor(bounds[e][1]/step+1e-8) for e in els]);
    if (lo<0).any() or (hi>scale).any() or (lo>hi).any() or lo.sum()>scale or hi.sum()<scale: raise ValueError(f'Composition bounds infeasible on {step:g} at.% grid / 成分范围与步长不兼容；choose a finer step or revise bounds. Grid lower sum={lo.sum()*step:g}, upper sum={hi.sum()*step:g}.')
    if lo.sum()==scale or hi.sum()==scale:
        return pd.DataFrame([lo if lo.sum()==scale else hi],columns=els)*step
    rows=set()
    for _ in range(n*25):
        a=lo.copy(); left=scale-a.sum(); order=rng.permutation(len(els))
        for p,i in enumerate(order):
            remain=order[p+1:]; low=max(0,left-(hi[remain]-lo[remain]).sum()); high=min(left,hi[i]-lo[i]); z=rng.integers(low,high+1); a[i]+=z; left-=z
        rows.add(tuple(a));
        if len(rows)>=n: break
    return pd.DataFrame(np.asarray(sorted(rows))*step,columns=els)

def inverse(df,results,els,desc,extras,raw,bounds,conditions,directions,step=1,pool_size=2000,iterations=60,seed=42,min_entropy=0,thresholds=None,categorical=None,process_mode='fixed',source_unit='at.%',process_ranges=None,process_categories=None):
    thresholds=thresholds or {}; categorical=[e for e in extras if process_type(df[e],e)=='categorical'] if categorical is None else categorical
    pool=pool_grid(els,bounds,step,pool_size,seed); x=pool[els].to_numpy()/100; ent=descriptors(x,els,['S_mix_J_molK']).iloc[:,0]; pool=pool.loc[ent>=min_entropy].reset_index(drop=True)
    if not len(pool): raise ValueError('没有满足构型熵约束的成分候选。')
    targets=list(directions); rng=np.random.default_rng(seed)
    if process_mode in ('observed','bounded') and extras:
        ranges=process_ranges or {}; categories=process_categories or {}
        combinations=constrained_process_pool(df,extras,categorical,process_mode,conditions,ranges,categories,results,targets)
        total=len(pool)*len(combinations)
        if process_mode=='bounded' and any(hi>lo for lo,hi in ranges.values()):
            # Sample independent continuous values only for explicitly bounded fields.
            ids=rng.choice(total,pool_size,replace=total<pool_size)
        else: ids=rng.choice(total,min(pool_size,total),replace=False)
        pool=pd.concat([pool.iloc[ids//len(combinations)].reset_index(drop=True),combinations.iloc[ids%len(combinations)].reset_index(drop=True)],axis=1)
        if process_mode=='bounded':
            for e,(lo,hi) in ranges.items(): pool[e]=rng.uniform(lo,hi,len(pool)) if hi>lo else float(lo)
            for e,v in conditions.items():
                if e not in categorical: pool[e]=float(v)
        pool=pool.drop_duplicates().reset_index(drop=True)
    else:
        for c in extras:
            pool[c]=str(conditions[c]).strip() if c in categorical else float(conditions[c])
            if c in categorical:
                for t in targets:
                    if c in results[t]['cols'] and pool[c].iloc[0] not in set(results[t]['train_x'][c].dropna()): raise ValueError(f'{c}: selected category absent from {t} training subset.')
    xx=features(pool,els,'at.%',desc,extras,raw,categorical); signs=np.array([1 if directions[t]=='max' else -1 for t in targets]); n=len(pool); z=pool[els].to_numpy()/100
    if process_mode in ('observed','bounded') and extras:
        for e in extras:
            if e in categorical: block=pd.get_dummies(pool[e],dtype=float).to_numpy()/np.sqrt(2)
            else:
                v=pool[e].to_numpy(float); block=((v-v.min())/max(np.ptp(v),1e-12))[:,None]
            z=np.column_stack([z,block])
    def evaluate(ids): return np.column_stack([results[t]['model'].predict(xx.iloc[ids][results[t]['cols']]).ravel() for t in targets])
    initial=min(n,max(8,2*len(els)),iterations); ids=list(rng.choice(n,initial,replace=False)); vals=evaluate(ids); history=[dict(iteration=i+1,pool_index=int(j),phase='initial') for i,j in enumerate(ids)]
    for it in range(initial,min(iterations,n)):
        remaining=np.setdiff1d(np.arange(n),ids); weights=rng.dirichlet(np.ones(len(targets))); normalized=(vals*signs-np.mean(vals*signs,axis=0))/np.maximum(np.std(vals,axis=0),1e-9); utility=normalized@weights
        for j,t in enumerate(targets):
            if t in thresholds: utility-=3*np.maximum(0,(float(thresholds[t])-vals[:,j])*signs[j])/max(np.std(vals[:,j]),1e-9)
        gp=GaussianProcessRegressor(kernel=Matern(length_scale=.2,nu=2.5),alpha=1e-5,normalize_y=True,optimizer=None,random_state=seed).fit(z[ids],utility)
        mu,sd=gp.predict(z[remaining],return_std=True); pick=int(remaining[np.argmax(mu+2*sd)]); ids.append(pick); vals=np.vstack([vals,evaluate([pick])]); history.append(dict(iteration=it+1,pool_index=pick,phase='GP-UCB',acquisition_std=float(sd[np.argmax(mu+2*sd)])))
    # Preserve the original BO search and its history; evaluate every remaining
    # feasible-pool candidate exactly once, in bounded batches. Align predictions
    # with the original pool indices before threshold and Pareto screening.
    bo_ids=np.asarray(ids,dtype=int)
    all_vals=np.empty((n,len(targets)),dtype=float); all_vals[bo_ids]=vals
    remaining=np.setdiff1d(np.arange(n),bo_ids)
    for start in range(0,len(remaining),512):
        batch=remaining[start:start+512]; all_vals[batch]=evaluate(batch)
    if not np.isfinite(all_vals).all():
        raise ValueError('Non-finite property predictions in candidate pool / 候选池存在非有限性能预测值，无法可靠筛选 Pareto 前沿。')
    vals=all_vals
    out=pool.copy().reset_index(drop=True); out['S_mix_J_molK']=descriptors(out[els].to_numpy()/100,els,['S_mix_J_molK']).iloc[:,0]
    for j,t in enumerate(targets): out['pred_'+t]=vals[:,j]
    feasible=np.ones(len(out),dtype=bool)
    for j,t in enumerate(targets):
        if t in thresholds: feasible &= (vals[:,j]-float(thresholds[t]))*signs[j]>=0
    out['Feasible']=feasible; front=np.zeros(len(out),dtype=bool)
    if feasible.any(): front[feasible]=pareto(vals[feasible]*signs)
    out['Pareto']=front; out['compromise_score']=np.nan
    if front.any():
        v=vals[front]*signs; span=np.ptp(v,axis=0); scaled=(v-v.min(axis=0))/np.where(span>0,span,1); scaled[:,span==0]=1; out.loc[front,'compromise_score']=np.sqrt(np.mean((1-scaled)**2,axis=1))
    out['BO_iteration']=np.nan; out.loc[bo_ids,'BO_iteration']=np.arange(1,len(bo_ids)+1)
    out['evaluation_stage']='full_pool'; out.loc[bo_ids,'evaluation_stage']='BO'
    out['pool_index']=np.arange(n)
    # composition novelty, not calibrated uncertainty
    ref=fractions(df,els,source_unit); out['nearest_composition_L2_atpct']=np.concatenate([pairwise_distances(out[els].iloc[start:start+512].to_numpy(),ref*100).min(1) for start in range(0,n,512)])
    return out,pd.DataFrame(history),len(pool)
