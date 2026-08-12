from pathlib import Path
import numpy as np


def main():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(500, 3))
    current = rng.normal(size=(500, 3))
    current[:, 0] += 1.2
    current[:, 2] *= 1.8
    out = Path(__file__).with_name("drift.npz")
    np.savez(out, reference=reference, current=current, feature_names=np.array(["mean_shift", "stable", "variance_shift"]))
    print(out)


if __name__ == "__main__":
    main()
