from pathlib import Path
import json, numpy as np
from shiftglass import drift_report

def main():
    rng=np.random.default_rng(44); n=1800
    ref=rng.normal(size=(n,5)); cur=rng.normal(size=(n,5))
    cur[:,1]=rng.normal(loc=1.35,scale=1.0,size=n)
    cur[:,4]=rng.normal(loc=0.0,scale=2.1,size=n)
    r=drift_report(ref,cur,["stable_a","mean_shift","stable_b","stable_c","variance_shift"],seed=7)
    top=[x.feature for x in r.features[:2]]
    result={"n":n,"top_two":top,"mmd2":round(r.mmd2,6),"covariance_shift":round(r.covariance_shift,6),"feature_scores":{x.feature:round(x.score,6) for x in r.features}}
    assert set(top)=={"mean_shift","variance_shift"}
    Path(__file__).with_name("results.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
