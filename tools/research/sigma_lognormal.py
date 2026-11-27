"""Sigma-Lognormal signature model for synthetic trajectory generation (Lai et al., AAAI 2020)."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.signal import butter, filtfilt

# Admissible parameter perturbation ranges (Table 1).
_ADMISSIBLE = {
    "D":     (-0.1000, 0.1000),
    "t0":    (-0.0825, 0.0850),
    "mu":    (-0.3950, 0.3775),
    "sigma": (-0.3250, 0.2875),
}

# Distortion bands for ranking: inner (G1) vs outer annulus (G2).
_LEVELS = {
    1: {"scale": {"D": 0.25, "t0": 0.25, "mu": 0.17, "sigma": 0.25},
        "annulus": None, "theta": 0.10},
    2: {"scale": {"D": 1.0, "t0": 1.0, "mu": 1.0, "sigma": 1.0},
        "annulus": (0.40, 0.65), "theta": 0.20},
}

# Matches config.settings.TEMPORAL_SAMPLE_RATE_HZ without importing project config.
_CUTOFF_HZ, _FILTER_ORDER, _SAMPLE_RATE_HZ = 10.0, 2, 100.0
_EPS = 1e-9


@dataclass(frozen=True)
class LognormalStroke:
    """One lognormal impulse response. Angles are radians."""
    D: float
    t0: float
    mu: float
    sigma: float
    theta_s: float = 0.0
    theta_e: float = 0.0


@dataclass(frozen=True)
class SignatureParameters:
    """A signature as a stroke list plus what is needed to rebuild it."""
    strokes: tuple
    duration: float
    length: int
    origin: tuple = (0.0, 0.0)


def stroke_velocity(t, stroke):
    """Compute one lognormal stroke speed profile (equation 1), zero for `t <= t0`."""
    t = np.asarray(t, dtype=np.float64)
    dt = t - stroke.t0
    out = np.zeros_like(dt)
    live = dt > _EPS
    if not np.any(live):
        return out
    d = dt[live]
    sigma = max(abs(stroke.sigma), 1e-3)
    out[live] = (stroke.D / (sigma * np.sqrt(2 * np.pi) * d)
                 * np.exp(-((np.log(d) - stroke.mu) ** 2) / (2 * sigma ** 2)))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth(v):
    """Butterworth low-pass, as the paper applies before parameter extraction."""
    padlen = 3 * _FILTER_ORDER
    if len(v) <= padlen + 1:
        return np.asarray(v, dtype=np.float64)
    b, a = butter(_FILTER_ORDER, _CUTOFF_HZ / (0.5 * _SAMPLE_RATE_HZ), btype="low")
    return filtfilt(b, a, np.asarray(v, dtype=np.float64))


def _extract_analytic(t, v, peak):
    """Analytically solve `(D, t0, mu, sigma)` for one velocity peak using its two inflection points."""
    n = len(v)
    if peak <= 1 or peak >= n - 2 or v[peak] <= _EPS:
        return None

    # Inflection points bracket the peak: nearest curvature sign changes either
    # side, on the smoothed residual.
    curvature = np.gradient(np.gradient(v))
    left = None
    for i in range(peak - 1, 0, -1):
        if curvature[i] > 0 >= curvature[i + 1]:
            left = i
            break
    right = None
    for i in range(peak + 1, n - 1):
        if curvature[i] > 0 >= curvature[i - 1]:
            right = i
            break
    if left is None or right is None or right <= peak <= left:
        return None

    tp, t1, t2 = t[peak], t[left], t[right]
    if not (t1 < tp < t2):
        return None

    def mismatch(t0):
        a = np.log(max(t2 - t0, _EPS)) - np.log(max(tp - t0, _EPS))
        b = np.log(max(tp - t0, _EPS)) - np.log(max(t1 - t0, _EPS))
        if b - a <= 1e-8:
            return 1e6
        sigma = np.sqrt(b - a)
        return (a + b) - sigma * np.sqrt(sigma ** 2 + 4.0)

    # t0 lies strictly before the first inflection; scan for a sign change and
    # refine with Brent. A bracket that never changes sign means this peak is
    # not a clean lognormal, so the stroke is rejected.
    span = max(t2 - t1, 1.0 / _SAMPLE_RATE_HZ)
    grid = t1 - np.geomspace(1e-3 * span, 40.0 * span, 60)
    values = np.array([mismatch(x) for x in grid])
    sign_change = np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0]
    if not len(sign_change):
        return None
    k = sign_change[0]
    try:
        t0 = brentq(mismatch, grid[k + 1], grid[k], xtol=1e-9, maxiter=120)
    except (ValueError, RuntimeError):
        return None

    a = np.log(max(t2 - t0, _EPS)) - np.log(max(tp - t0, _EPS))
    b = np.log(max(tp - t0, _EPS)) - np.log(max(t1 - t0, _EPS))
    if b - a <= 1e-8:
        return None
    sigma = float(np.sqrt(b - a))
    if not (0.05 <= sigma <= 1.2):
        return None
    mu = float(np.log(max(tp - t0, _EPS)) + sigma ** 2)
    D = float(v[peak] * sigma * np.sqrt(2 * np.pi) * np.exp(mu - sigma ** 2 / 2))
    if not np.isfinite(D) or D <= 0:
        return None
    return D, float(t0), mu, sigma


def _seed_from_peak(t, v, peak):
    """Estimate initial stroke parameters from half-height peak width when analytic extraction fails."""
    n = len(v)
    if peak <= 0 or peak >= n - 1 or v[peak] <= _EPS:
        return None

    half = 0.5 * v[peak]
    left = peak
    while left > 0 and v[left] > half:
        left -= 1
    right = peak
    while right < n - 1 and v[right] > half:
        right += 1

    width = max(t[right] - t[left], 2.0 / _SAMPLE_RATE_HZ)
    sigma = 0.35
    # Onset roughly one width before the peak keeps t0 < t_p, which the model
    # requires, without pushing the lognormal's mode far from where it was seen.
    lag = max(width, 1.0 / _SAMPLE_RATE_HZ)
    t0 = t[peak] - lag
    mu = float(np.log(lag) + sigma ** 2)
    D = float(v[peak] * sigma * np.sqrt(2 * np.pi) * np.exp(mu - sigma ** 2 / 2))
    if not np.isfinite(D) or D <= 0:
        return None
    return D, float(t0), mu, sigma


def decompose(trajectory):
    """Fit a sum of lognormal strokes to a `[L, >=2]` trajectory via greedy extraction and least squares."""
    xy = np.asarray(trajectory, dtype=np.float64)[:, :2]
    n = len(xy)
    if n < 8:
        return SignatureParameters(strokes=(), duration=max(n / _SAMPLE_RATE_HZ, _EPS),
                                   length=n, origin=tuple(xy[0]) if n else (0.0, 0.0))

    t = np.arange(n) / _SAMPLE_RATE_HZ
    duration = t[-1]
    vx = np.gradient(xy[:, 0]) * _SAMPLE_RATE_HZ
    vy = np.gradient(xy[:, 1]) * _SAMPLE_RATE_HZ
    speed = _smooth(np.hypot(vx, vy))

    if speed.max() < _EPS:
        return SignatureParameters(strokes=(), duration=duration, length=n,
                                   origin=tuple(xy[0]))

    # Greedily extract peaks on the residual speed profile so overlapping strokes separate cleanly.
    heading = np.unwrap(np.arctan2(vy, vx))
    vx_s, vy_s = _smooth(vx), _smooth(vy)
    residual = speed.copy()
    max_strokes = max(4, min(24, int(round(duration / 0.06))))
    floor = 0.02 * speed.max()

    seeds6, lower6, upper6 = [], [], []
    for _ in range(max_strokes):
        if residual.max() <= floor:
            break
        p = int(np.argmax(residual))
        found = _extract_analytic(t, residual, p)
        if found is None:
            # Overlapping peaks can skew left and fail exact lognormal solvability; fall back to width seed.
            found = _seed_from_peak(t, residual, p)
            if found is None:
                lo, hi = max(p - 2, 0), min(p + 3, n)
                residual[lo:hi] = 0.0
                continue

        D, t0, mu, sigma = found
        contribution = stroke_velocity(t, LognormalStroke(D, t0, mu, sigma))
        residual = np.maximum(residual - contribution, 0.0)

        # Angles come from the trajectory heading where this stroke is active,
        # weighted by its own speed so overlapping strokes do not borrow each
        # other's direction.
        weight = contribution / max(contribution.max(), _EPS)
        active = np.where(weight > 0.25)[0]
        if len(active):
            h_s, h_e = heading[active[0]], heading[active[-1]]
        else:
            h_s = h_e = heading[min(p, n - 1)]

        seeds6 += [D, t0, mu, sigma, h_s, h_e]
        lower6 += [1e-6, -duration, -8.0, 0.05, h_s - np.pi, h_e - np.pi]
        upper6 += [50.0, max(t[p], 1.0 / _SAMPLE_RATE_HZ), 2.0, 1.2,
                   h_s + np.pi, h_e + np.pi]

    if not seeds6:
        return SignatureParameters(strokes=(), duration=duration, length=n,
                                   origin=tuple(xy[0]))

    def _velocity(flat):
        mx, my = np.zeros(n), np.zeros(n)
        for i in range(0, len(flat), 6):
            s = LognormalStroke(*flat[i:i + 6])
            v = stroke_velocity(t, s)
            travelled = np.cumsum(v) / _SAMPLE_RATE_HZ
            phi = s.theta_s + (s.theta_e - s.theta_s) * travelled / max(abs(s.D), _EPS)
            mx += v * np.cos(phi)
            my += v * np.sin(phi)
        return mx, my

    def residual(flat):
        mx, my = _velocity(flat)
        return np.concatenate([mx - vx_s, my - vy_s])

    # Seeds are derived from measured peaks, so a degenerate signature can push
    # one just past its bound and least_squares rejects the whole fit. Clamp
    # rather than widen the bounds: the bounds are what keep the fitted strokes
    # physically meaningful.
    lower6, upper6 = np.array(lower6), np.array(upper6)
    start = np.clip(np.array(seeds6), lower6, upper6)

    fit = least_squares(residual, start, bounds=(lower6, upper6),
                        method="trf", max_nfev=120, ftol=1e-7, xtol=1e-7, x_scale="jac")

    strokes = [LognormalStroke(*fit.x[i:i + 6]) for i in range(0, len(fit.x), 6)]

    return SignatureParameters(strokes=tuple(strokes), duration=duration,
                               length=n, origin=tuple(xy[0]))


def reconstruct(params):
    """Integrate summed stroke velocities (equations 3 to 5) into `[length, 2]` coordinates."""
    n = params.length
    out = np.zeros((n, 2), dtype=np.float32)
    out[:] = np.asarray(params.origin, dtype=np.float32)
    if not params.strokes or n < 2:
        return out

    t = np.arange(n) / _SAMPLE_RATE_HZ
    vx = np.zeros(n)
    vy = np.zeros(n)
    for s in params.strokes:
        v = stroke_velocity(t, s)
        # phi turns linearly with distance covered, so the angle is driven by
        # the cumulative integral of this stroke's own speed (equation 2).
        travelled = np.cumsum(v) / _SAMPLE_RATE_HZ
        phi = s.theta_s + (s.theta_e - s.theta_s) * travelled / max(abs(s.D), _EPS)
        vx += v * np.cos(phi)
        vy += v * np.sin(phi)

    out[:, 0] = params.origin[0] + np.cumsum(vx) / _SAMPLE_RATE_HZ
    out[:, 1] = params.origin[1] + np.cumsum(vy) / _SAMPLE_RATE_HZ
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _draw(rng, low, high, scale, annulus):
    """Sample a scaled perturbation uniformly or from a signed inner/outer annulus."""
    if annulus is None:
        return rng.uniform(scale * low, scale * high)
    inner, outer = annulus
    return (rng.uniform(scale * outer * low, scale * inner * low)
            if rng.random() < 0.5
            else rng.uniform(scale * inner * high, scale * outer * high))


def distort(params, level, rng):
    """Perturb all strokes coherently with one signature-level random draw (equations 6 to 11)."""
    if level not in _LEVELS:
        raise ValueError(f"unknown distortion level {level!r}; expected 1 or 2")
    spec = _LEVELS[level]

    r = {k: _draw(rng, *_ADMISSIBLE[k], spec["scale"][k], spec["annulus"])
         for k in _ADMISSIBLE}
    r_ts = rng.uniform(-spec["theta"], spec["theta"])
    r_te = rng.uniform(-spec["theta"], spec["theta"])

    strokes = tuple(
        replace(s,
                D=s.D * (1 + r["D"]),
                t0=s.t0 * (1 + r["t0"]),
                mu=s.mu * (1 + r["mu"]),
                sigma=s.sigma * (1 + r["sigma"]),
                theta_s=s.theta_s + r_ts,
                theta_e=s.theta_e + r_te)
        for s in params.strokes)
    return replace(params, strokes=strokes)


def synthesize(trajectory, n_low=5, n_median=10, seed=0):
    """Return ranked synthetic pools `(G1, G2)` of `[L, 2]` float32 trajectories from one template."""
    rng = np.random.default_rng(seed)
    params = decompose(trajectory)
    return ([reconstruct(distort(params, 1, rng)) for _ in range(n_low)],
            [reconstruct(distort(params, 2, rng)) for _ in range(n_median)])
