"""GeoLife GPS Trajectories 1.3: parsing, cleaning, local projection, segmentation, sparse-observation tasks and the
GPS <-> model-unit coordinate transform. Pure data handling; nothing here touches a model.

Dataset licence: Microsoft Research License Agreement (non-commercial use only, no redistribution of the data or
derivative works). Cite Zheng et al. (WWW 2009; UbiComp 2008; IEEE Data Eng. Bull. 2010) when using it.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# WGS-84
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
PLT_HEADER_LINES = 6
EXCEL_EPOCH_UNIX_DAYS = 25569.0          # 1899-12-30 -> 1970-01-01 in days


# ----------------------------------------------------------------------------------------------------------- parsing
@dataclass
class ParsedPLT:
    lat: np.ndarray
    lon: np.ndarray
    t_unix: np.ndarray                  # int64 seconds, GMT, from the date+time strings (fields 6-7)
    n_lines: int                        # data lines after the 6 header lines
    n_malformed: int                    # wrong field count or non-numeric lat/lon/date
    n_invalid_coord: int                # out of range, non-finite or (0, 0)
    n_datenum_mismatch: int             # |field 5 - fields 6/7| > 1 s
    header_ok: bool


def parse_plt(raw: bytes) -> ParsedPLT:
    """Parse one GeoLife .plt file. Malformed and invalid-coordinate rows are dropped and counted; nothing is sorted."""
    lines = raw.decode('utf-8', errors='replace').splitlines()
    header_ok = len(lines) >= PLT_HEADER_LINES and lines[0].strip().lower().startswith('geolife trajectory')
    body = [ln for ln in lines[PLT_HEADER_LINES:] if ln.strip()]
    lat, lon, dnum, stamp = [], [], [], []
    malformed = 0
    for ln in body:
        f = ln.strip().split(',')
        if len(f) != 7:
            malformed += 1
            continue
        try:
            la, lo, dn = float(f[0]), float(f[1]), float(f[4])
        except ValueError:
            malformed += 1
            continue
        lat.append(la); lon.append(lo); dnum.append(dn); stamp.append(f[5].strip() + 'T' + f[6].strip())
    lat = np.asarray(lat, dtype=np.float64); lon = np.asarray(lon, dtype=np.float64); dnum = np.asarray(dnum, dtype=np.float64)
    try:
        t_unix = np.asarray(stamp, dtype='datetime64[s]').astype(np.int64)
    except ValueError:                                   # some unparseable date string: fall back row by row
        t_unix = np.empty(len(stamp), dtype=np.int64); bad = np.zeros(len(stamp), bool)
        for i, s in enumerate(stamp):
            try:
                t_unix[i] = np.datetime64(s, 's').astype(np.int64)
            except ValueError:
                bad[i] = True
        malformed += int(bad.sum())
        keep = ~bad
        lat, lon, dnum, t_unix = lat[keep], lon[keep], dnum[keep], t_unix[keep]
    mismatch = int(np.sum(np.abs((dnum - EXCEL_EPOCH_UNIX_DAYS) * 86400.0 - t_unix) > 1.0))
    valid = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180) & ~((lat == 0) & (lon == 0)))
    return ParsedPLT(lat[valid], lon[valid], t_unix[valid], n_lines=len(body), n_malformed=malformed,
                     n_invalid_coord=int((~valid).sum()), n_datenum_mismatch=mismatch, header_ok=header_ok)


def clean_time_order(t_unix: np.ndarray):
    """Indices of records kept by the time-order policy: a record is kept only if its timestamp is strictly later than
    the last kept one (first of duplicates kept; backward jumps dropped). Returns (keep_idx, n_duplicate, n_backward)."""
    keep = np.zeros(len(t_unix), dtype=bool)
    last = None; dup = back = 0
    for i, t in enumerate(t_unix):
        if last is None or t > last:
            keep[i] = True; last = t
        elif t == last:
            dup += 1
        else:
            back += 1
    return np.flatnonzero(keep), dup, back


# ------------------------------------------------------------------------------------------------------- geodesy
def gaussian_radius(lat0_deg: float) -> float:
    """Gaussian mean radius of curvature sqrt(M N) of the WGS-84 ellipsoid at latitude lat0."""
    s2 = np.sin(np.radians(lat0_deg)) ** 2
    w = np.sqrt(1 - WGS84_E2 * s2)
    m = WGS84_A * (1 - WGS84_E2) / w ** 3
    n = WGS84_A / w
    return float(np.sqrt(m * n))


def haversine(lat1, lon1, lat2, lon2, radius: float = 6371008.8):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - np.asarray(lon1)) / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def aeqd_forward(lat, lon, lat0: float, lon0: float, radius: float | None = None):
    """Spherical azimuthal equidistant projection centred at (lat0, lon0), sphere radius = local Gaussian radius of the
    WGS-84 ellipsoid. Returns x (east, m), y (north, m). Distances and azimuths from the origin are exact on that sphere."""
    r = gaussian_radius(lat0) if radius is None else radius
    p, l = np.radians(lat), np.radians(lon)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    dl = l - l0
    cos_c = np.clip(np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(dl), -1, 1)
    c = np.arccos(cos_c)
    with np.errstate(invalid='ignore', divide='ignore'):
        k = np.where(c < 1e-12, 1.0, c / np.sin(c))
    x = r * k * np.cos(p) * np.sin(dl)
    y = r * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl))
    return x, y


def aeqd_inverse(x, y, lat0: float, lon0: float, radius: float | None = None):
    r = gaussian_radius(lat0) if radius is None else radius
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    rho = np.hypot(x, y)
    c = rho / r
    with np.errstate(invalid='ignore', divide='ignore'):
        lat = np.where(rho < 1e-9, p0, np.arcsin(np.cos(c) * np.sin(p0) + np.where(rho < 1e-9, 0, y * np.sin(c) * np.cos(p0) / rho)))
    lon = l0 + np.arctan2(x * np.sin(c), rho * np.cos(p0) * np.cos(c) - y * np.sin(p0) * np.sin(c))
    return np.degrees(lat), np.degrees(lon)


# ------------------------------------------------------------------------------------------------- segmentation
@dataclass
class SegmentConfig:
    max_gap_s: float = 60.0              # a larger time gap between consecutive records splits a run
    max_speed_mps: float = 70.0          # a larger step speed (implausible jump) splits a run
    window_points: int = 200             # consecutive cleaned records per evaluation segment
    min_extent_m: float = 100.0          # max distance of any record from the segment start (excludes stationary jitter)
    min_context_points: int = 60         # records with t_model <= 7 (30 observed + >= 30 hidden)
    min_future_points: int = 10          # records with 7 < t_model <= 10


def continuous_runs(t_unix, lat, lon, cfg: SegmentConfig):
    """Split a cleaned (strictly increasing time) record sequence into runs without large gaps or implausible jumps.
    Returns a list of (start, stop) index pairs (stop exclusive)."""
    n = len(t_unix)
    if n < 2:
        return [(0, n)] if n else []
    dt = np.diff(t_unix).astype(np.float64)
    step = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    brk = (dt > cfg.max_gap_s) | (step / np.maximum(dt, 1e-9) > cfg.max_speed_mps)
    cuts = np.flatnonzero(brk) + 1
    edges = np.concatenate([[0], cuts, [n]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def model_time_full(t_unix_window):
    """Protocol A (primary): the whole segment duration is mapped onto [0, 10]."""
    t = np.asarray(t_unix_window, dtype=np.float64)
    return 10.0 * (t - t[0]) / (t[-1] - t[0])


def window_checks(t_unix, lat, lon, cfg: SegmentConfig):
    """Objective eligibility checks for one window (independent of any model output). Returns (ok, reason)."""
    if len(t_unix) != cfg.window_points:
        return False, 'too_short'
    x, y = aeqd_forward(lat, lon, lat[0], lon[0])
    if np.max(np.hypot(x, y)) < cfg.min_extent_m:
        return False, 'stationary'
    tm = model_time_full(t_unix)
    if np.sum(tm <= 7.0) < cfg.min_context_points:
        return False, 'few_context_points'
    if np.sum(tm > 7.0) < cfg.min_future_points:
        return False, 'few_future_points'
    return True, 'ok'


def candidate_windows(t_unix, lat, lon, cfg: SegmentConfig):
    """Non-overlapping windows of ``window_points`` consecutive records from each continuous run (stride = window
    length, starting at the run start). Returns list of (start, stop, ok, reason) in cleaned-record indices."""
    out = []
    for a, b in continuous_runs(t_unix, lat, lon, cfg):
        for s in range(a, b - cfg.window_points + 1, cfg.window_points):
            e = s + cfg.window_points
            ok, why = window_checks(t_unix[s:e], lat[s:e], lon[s:e], cfg)
            out.append((s, e, ok, why))
    return out


# -------------------------------------------------------------------------------------------------------- archive
class GeoLifeArchive:
    """Read-only access to the original zip (never extracted or modified)."""

    def __init__(self, zip_path):
        self.path = Path(zip_path)
        self.zf = zipfile.ZipFile(self.path)
        self.plt_names = sorted(n for n in self.zf.namelist() if n.lower().endswith('.plt'))
        self.label_names = sorted(n for n in self.zf.namelist() if n.endswith('labels.txt'))

    @staticmethod
    def user_of(name: str) -> str:
        parts = Path(name).parts
        return parts[parts.index('Data') + 1]

    def read(self, name: str) -> bytes:
        return self.zf.read(name)

    def load_clean(self, name: str):
        """Parsed, coordinate-valid, time-ordered records of one trajectory file."""
        p = parse_plt(self.read(name))
        keep, _, _ = clean_time_order(p.t_unix)
        return p.t_unix[keep], p.lat[keep], p.lon[keep]

    def labels(self, user: str):
        """Transportation-mode labels (GMT) of one user, as (start_unix, end_unix, mode) arrays; empty if unlabelled."""
        name = next((n for n in self.label_names if self.user_of(n) == user), None)
        if name is None:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, object)
        rows = [ln.split('\t') for ln in self.read(name).decode().splitlines()[1:] if ln.strip()]
        st = np.array([np.datetime64(r[0].replace('/', '-').replace(' ', 'T'), 's') for r in rows]).astype(np.int64)
        en = np.array([np.datetime64(r[1].replace('/', '-').replace(' ', 'T'), 's') for r in rows]).astype(np.int64)
        return st, en, np.array([r[2].strip() for r in rows], dtype=object)


def file_sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def label_for_window(labels, t0: int, t1: int):
    """Transport mode whose labelled interval fully contains [t0, t1], else None."""
    st, en, md = labels
    hit = np.flatnonzero((st <= t0) & (en >= t1))
    return str(md[hit[0]]) if len(hit) else None


# ------------------------------------------------------------------------------------------- interpolation tasks
@dataclass
class Task:
    """One sparse-observation problem built from a real GPS segment. Coordinates are local metres (AEQD at the
    segment's first record, which is always an observation)."""
    t_model: np.ndarray                  # all records, protocol A time in [0, 10]
    t_sec: np.ndarray                    # elapsed seconds from the segment start
    xy_m: np.ndarray                     # [n, 2] metres
    obs_idx: np.ndarray                  # observed records (sorted, includes 0, all with t_model <= 7)
    interp_idx: np.ndarray               # withheld context records inside [first obs, last obs]: interpolation targets
    ctx_tail_idx: np.ndarray             # withheld context records after the last observation (<= 7): reported, not scored
    future_idx: np.ndarray               # records with 7 < t_model <= 10: forecast targets (secondary experiment)
    meta: dict = field(default_factory=dict)


def sample_observations(t_model, n_obs: int, rng: np.random.Generator, context_end: float = 7.0):
    """Exactly ``n_obs`` observed records from the context interval: the first record (t = 0, as in every training
    trajectory) plus n_obs - 1 drawn uniformly without replacement from the other context records. Uses only times."""
    ctx = np.flatnonzero(np.asarray(t_model) <= context_end + 1e-12)
    if ctx.size < n_obs + 1 or ctx[0] != 0:
        raise ValueError(f'context has {ctx.size} records; need at least {n_obs + 1} including t=0')
    rest = rng.choice(ctx[1:], size=n_obs - 1, replace=False)
    return np.sort(np.concatenate([[0], rest]))


def build_task(t_unix, lat, lon, n_obs: int, seed: int, context_end: float = 7.0, meta=None) -> Task:
    t_model = model_time_full(t_unix)
    t_sec = (np.asarray(t_unix, np.float64) - float(t_unix[0]))
    x, y = aeqd_forward(lat, lon, float(lat[0]), float(lon[0]))
    xy = np.stack([x, y], 1)
    obs = sample_observations(t_model, n_obs, np.random.default_rng(seed), context_end)
    ctx = np.flatnonzero(t_model <= context_end + 1e-12)
    hidden = np.setdiff1d(ctx, obs)
    interp = hidden[hidden < obs[-1]]
    tail = hidden[hidden > obs[-1]]
    future = np.flatnonzero((t_model > context_end + 1e-12) & (t_model <= 10.0 + 1e-12))
    m = dict(meta or {}); m.update(origin_lat=float(lat[0]), origin_lon=float(lon[0]), seed=int(seed), n_obs=int(n_obs))
    return Task(t_model, t_sec, xy, obs, interp, tail, future, m)


# ---------------------------------------------------------------------------- GPS metres <-> model coordinate units
@dataclass(frozen=True)
class NormStrategy:
    """Per-trajectory similarity transform from local metres to the synthetic model's raw coordinate units, computed
    from the OBSERVED records only. model = R(theta) (xy_m - center) / s * target + offset."""
    name: str
    center: str = 'obs_mean'             # 'obs_mean' | 'first_obs'
    spread: str = 'obs_rms'              # 'obs_rms' (RMS distance about the centre) | 'obs_max' (max distance)
    target: float = 1.0                  # model-unit size the spread is mapped to
    rotate: float = 0.0                  # radians, applied after centring (sensitivity analysis only)
    offset: tuple = (0.0, 0.0)           # model-unit translation added at the end (sensitivity analysis only)


@dataclass
class Transform:
    center: np.ndarray
    s: float                             # metres per (target) unit before scaling
    target: float
    rot: float
    offset: np.ndarray

    def to_model(self, xy_m):
        c, s_ = np.cos(self.rot), np.sin(self.rot)
        R = np.array([[c, -s_], [s_, c]])
        return ((np.asarray(xy_m, np.float64) - self.center) @ R.T) / self.s * self.target + self.offset

    def to_metres(self, xy_model):
        c, s_ = np.cos(self.rot), np.sin(self.rot)
        R = np.array([[c, -s_], [s_, c]])
        return ((np.asarray(xy_model, np.float64) - self.offset) * self.s / self.target) @ R + self.center

    @property
    def metres_per_model_unit(self) -> float:
        return self.s / self.target


def fit_transform(xy_obs_m, strategy: NormStrategy) -> Transform:
    """Fit the transform on the observed coordinates only (no withheld or future record can influence it)."""
    xy_obs_m = np.asarray(xy_obs_m, np.float64)
    center = xy_obs_m.mean(0) if strategy.center == 'obs_mean' else xy_obs_m[0].copy()
    d = np.linalg.norm(xy_obs_m - center, axis=1)
    s = float(np.sqrt(np.mean(d ** 2))) if strategy.spread == 'obs_rms' else float(d.max())
    s = max(s, 1e-6)
    return Transform(center, s, float(strategy.target), float(strategy.rotate), np.asarray(strategy.offset, np.float64))


# Synthetic three-shape TRAINING split, context (t <= 7) of the clean trajectories: per-trajectory RMS distance about
# the trajectory mean has median 1.005 (5th/95th percentile 0.572 / 1.665); per-trajectory maximum distance was
# measured the same way in scripts/21 (see results/geolife/tables/normalization_strategies.json).
DEFAULT_STRATEGY = NormStrategy('S1_obsmean_rms_1.0', 'obs_mean', 'obs_rms', 1.0)
