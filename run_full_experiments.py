# Usage: put this file next to leakage_and_temporal_experiments.py, then: python run_full_experiments.py creditcard.csv
# Writes full_results.jsonl incrementally (safe to stop and resume by reading the file). Order: seed 42 everything first, then seeds 1-4, then SMOTE for seeds 1-2.
import sys, pandas as pd, numpy as np, json
import leakage_and_temporal_experiments as exp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
df0=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'creditcard.csv')
df=exp.add_time(df0)
OUT='full_results.jsonl'
def log(seed,tag,model,r):
    with open(OUT,'a') as f: f.write(json.dumps(dict(seed=seed,scenario=tag,model=model,**r))+"\n")
SEEDS=[42,1,2,3,4]; ALL=["LR","DT","RF"]
def M(seed): return exp.models(seed)
def split(seed):
    i,j=train_test_split(np.arange(len(df)),test_size=0.2,stratify=df.Class,random_state=seed)
    return df.iloc[i],df.iloc[j]
def pass1(seed):
    tr,te=split(seed); ytr,yte=tr.Class,te.Class
    p=exp.prep_fit(tr); XA,XAt=exp.prep_apply(tr,p),exp.prep_apply(te,p)
    for n,m in M(seed).items(): log(seed,"A. correct",n,exp.metrics(m,XA,ytr,XAt,yte))
    pl=exp.prep_fit(df); XB,XBt=exp.prep_apply(tr,pl),exp.prep_apply(te,pl)
    for n,m in M(seed).items(): log(seed,"B. LEAKY scaler/cap before split",n,exp.metrics(m,XB,ytr,XBt,yte))
    cl=exp.top_k_by_separability(exp.prep_apply(df,pl),df.Class); co=exp.top_k_by_separability(XA,ytr)
    for tag,c in [("C0. feature sel. train only",co),("C1. LEAKY feature sel. all labels",cl)]:
        for n,m in M(seed).items(): log(seed,tag,n,exp.metrics(m,XA[c],ytr,XAt[c],yte))
    ds=df.sort_values("Time").reset_index(drop=True)
    for lab,mask in {"T1 day1->day2":ds.Time<86400,"T2 first36h->last12h":ds.Time<129600}.items():
        t,s=ds[mask],ds[~mask]; pp=exp.prep_fit(t,tcols=())
        for n,m in M(seed).items(): log(seed,"TEMPORAL "+lab,n,exp.metrics(m,exp.prep_apply(t,pp),t.Class,exp.prep_apply(s,pp),s.Class))
        ra,rb=train_test_split(ds,train_size=len(t)/len(ds),stratify=ds.Class,random_state=seed); pp=exp.prep_fit(ra,tcols=())
        for n,m in M(seed).items(): log(seed,"RANDOM ref for "+lab,n,exp.metrics(m,exp.prep_apply(ra,pp),ra.Class,exp.prep_apply(rb,pp),rb.Class))
    log(seed,"PASS1 DONE","-",{})
def pass2(seed):
    tr,te=split(seed); ytr,yte=tr.Class,te.Class
    p=exp.prep_fit(tr); XA,XAt=exp.prep_apply(tr,p),exp.prep_apply(te,p)
    Xe,ye=SMOTE(random_state=seed).fit_resample(XA,ytr)
    for n,m in M(seed).items(): log(seed,"E. correct SMOTE (train only)",n,exp.metrics(m,Xe,ye,XAt,yte))
    pl=exp.prep_fit(df)
    Xs,ys=SMOTE(random_state=seed).fit_resample(exp.prep_apply(df,pl),df.Class)
    a,b,c,d=train_test_split(Xs,ys,test_size=0.2,stratify=ys,random_state=seed)
    for n,m in M(seed).items(): log(seed,"D. LEAKY SMOTE before split",n,exp.metrics(m,a,c,b,d))
    log(seed,"PASS2 DONE","-",{})

for sd,fn in [(42,pass1),(42,pass2),(1,pass1),(2,pass1),(3,pass1),(4,pass1),(1,pass2),(2,pass2)]:
    fn(sd)
