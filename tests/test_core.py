import numpy as np
import pytest
from shiftglass import drift_report, rbf_mmd2


def test_identical_samples_have_small_mmd():
    rng=np.random.default_rng(1); X=rng.normal(size=(150,3))
    assert rbf_mmd2(X,X,seed=3) < 1e-10


def test_known_shifted_feature_ranks_first():
    rng=np.random.default_rng(2); X=rng.normal(size=(700,4)); Y=X.copy(); Y[:,2]=rng.normal(loc=2.0,scale=1.0,size=len(Y))
    r=drift_report(X,Y,["a","b","shifted","d"],seed=4)
    assert r.features[0].feature == "shifted"
    assert r.features[0].ks > .4


def test_variance_shift_is_detected():
    rng=np.random.default_rng(5); X=rng.normal(size=(800,2)); Y=X.copy(); Y[:,1]*=2.5
    r=drift_report(X,Y)
    f=next(x for x in r.features if x.feature=="x1")
    assert f.std_ratio > 2


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        drift_report(np.zeros((4,2)), np.zeros((4,3)))
