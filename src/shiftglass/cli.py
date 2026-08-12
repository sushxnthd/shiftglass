import argparse, json
import numpy as np
from .core import drift_report

def main():
    ap=argparse.ArgumentParser(description="Compare reference and current numerical datasets")
    ap.add_argument("npz", help="NPZ with reference and current arrays")
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args(); d=np.load(args.npz, allow_pickle=False)
    names=[str(x) for x in d["feature_names"]] if "feature_names" in d else None
    r=drift_report(d["reference"], d["current"], names, seed=args.seed)
    print(json.dumps({"mmd2":r.mmd2,"covariance_shift":r.covariance_shift,"features":[x.__dict__ for x in r.features]}, indent=2))

if __name__ == "__main__": main()
