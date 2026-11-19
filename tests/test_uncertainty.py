import numpy as np

from models import uncertainty as U

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]


def test_rearrange_sorts_and_clips():
    Q = np.array([[3.0, 1.0, -2.0, 5.0, 4.0, 6.0, 8.0, 7.0]])
    R = U.rearrange(Q)
    assert (np.diff(R, axis=1) >= 0).all() and R.min() >= 0


def test_prob_exceed_monotone_in_threshold():
    Q = np.array([[0, 1, 3, 6, 10, 15, 20, 26]], dtype=float)
    ps = [U.prob_exceed(Q, TAUS, t)[0] for t in (0.5, 5, 10, 20, 30, 50)]
    assert all(a >= b for a, b in zip(ps, ps[1:]))
    assert abs(U.prob_exceed(Q, TAUS, 6.0)[0] - 0.5) < 1e-6      # median
    assert 0 < U.prob_exceed(Q, TAUS, 30)[0] < 0.02               # tail beyond q98


def test_conformal_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y = rng.gamma(2, 3, size=4000)
    Q = np.tile(np.quantile(y, TAUS) * 0.7, (len(y), 1))          # too narrow on purpose
    reg = np.zeros(len(y), dtype=int)
    offs = U.conformal_offsets(Q, y, reg, TAUS)
    Q2 = U.apply_conformal(Q, reg, offs, TAUS)
    cov = np.mean((y >= Q2[:, 0]) & (y <= Q2[:, 6]))
    assert cov >= 0.89


def test_confidence_independent_labels():
    s1, l1, _ = U.confidence(3, 2.0, 30, 0, True)
    s2, l2, r2 = U.confidence(72, 40.0, 30, 10, False)
    assert l1 == "high" and l2 == "low" and s1 > s2 and r2
