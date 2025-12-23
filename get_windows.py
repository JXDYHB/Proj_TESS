# transit_window.py
import numpy as np
from scipy.signal import find_peaks

import numpy as np

def cut_transit_window(
    time,
    flux,
    t_center,
    window=0.5,
    min_consecutive=3,
    dip_sigma=3.0,
    inner_frac=0.5,
):
    """
    Extract and vet a candidate transit window centered at `t_center`.

    Adds a simple "consecutive dip points" requirement to suppress single-point outliers:
    within the central region |t| < inner_frac*window, require at least `min_consecutive`
    consecutive points with flux < baseline - dip_sigma*sigma.

    Parameters
    ----------
    time, flux : array_like
        Time and flux arrays.
    t_center : float
        Candidate center time.
    window : float
        Half-window size around t_center.
    min_consecutive : int
        Minimum number of consecutive in-transit-like points required.
    dip_sigma : float
        Dip threshold in units of sigma below baseline.
    inner_frac : float
        Only enforce the consecutive dip test within |t| < inner_frac*window.

    Returns
    -------
    dict or None
    """
    # ------------------
    # basic cut
    # ------------------
    mask = np.abs(time - t_center) < window
    t = np.asarray(time[mask], float)
    f_raw = np.asarray(flux[mask], float)

    if len(t) < 30:
        return None

    baseline = np.median(f_raw)
    if not np.isfinite(baseline) or baseline == 0:
        return None

    f = f_raw / baseline
    t = t - t_center

    # sort by time (important for "consecutive" logic)
    sidx = np.argsort(t)
    t = t[sidx]
    f = f[sidx]

    # ------------------
    # noise estimate (from edges)
    # ------------------
    abs_t = np.abs(t)
    edge_mask = abs_t > 0.7 * abs_t.max()
    if np.sum(edge_mask) < 10:
        return None

    f_edge = f[edge_mask]
    mad = np.median(np.abs(f_edge - np.median(f_edge)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(f_edge)

    if not np.isfinite(sigma) or sigma <= 0:
        return None

    # ------------------
    # NEW: consecutive dip requirement (Method 1)
    # ------------------
    inner_mask = abs_t < (inner_frac * window)
    if np.sum(inner_mask) < max(10, min_consecutive):
        return None

    base = np.median(f_edge)
    thresh = base - dip_sigma * sigma
    dip = (f < thresh) & inner_mask

    # max run length of consecutive True in `dip`
    max_run = 0
    run = 0
    for v in dip:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0

    if max_run < min_consecutive:
        return None

    # ------------------
    # depth SNR (now less likely to be single-point-driven)
    # ------------------
    depth = base - np.min(f)
    snr = depth / sigma
    if snr < 5.0:
        return None

    # ------------------
    # symmetry check
    # ------------------
    tau_pos = t[t > 0]
    tau_neg = t[t < 0]
    if len(tau_pos) < 10 or len(tau_neg) < 10:
        return None

    tau_max = min(tau_pos.max(), -tau_neg.min())
    tau_grid = np.linspace(0, tau_max, 200)

    f_r = np.interp(tau_grid,  t, f)
    f_l = np.interp(-tau_grid, t, f)

    mse = np.mean((f_r - f_l) ** 2)
    score = np.exp(-mse / (2 * sigma * sigma))

    if score < 0.6:
        return None

    e = np.ones_like(f) * sigma
    return dict(
        t=t,
        f=f,
        e=e,
        t_center=float(t_center),
        score=float(score),
        snr=float(snr),
    )



def collect_windows(sectors, sector_data, tic=120, window=0.5, max_windows_per_sector=2):
    """
    sector_data: dict-like, key=(sector, tic), value=LightCurve
    """
    windows = []

    for sector in sectors:
        print(f"\n--- Processing Sector {sector} ---")
        lc = sector_data[(sector, tic)]

        time = np.asarray(lc.time.value, float)
        flux = np.asarray(lc.flux.value, float)

        good = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[good], flux[good]

        if len(time) < 100:
            continue

        # clip outliers
        lo, hi = np.nanpercentile(flux, [0.1, 99.9])
        flux = np.clip(flux, lo, hi)

        # find dips
        inv_flux = 1.0 - flux
        sigma = np.std(inv_flux)
        dist_min = min(1000, len(inv_flux) // 2)

        peaks, props = find_peaks(
            inv_flux,
            prominence=3 * sigma,
            distance=dist_min
        )

        if len(peaks) == 0:
            continue

        # strongest dips first
        order = np.argsort(props["prominences"])[::-1]
        peaks = peaks[order][:max_windows_per_sector]

        for idx in peaks:
            win = cut_transit_window(time, flux, time[idx], window)
            if win is not None:
                win["sector"] = sector
                windows.append(win)

    print(f"\nTotal accepted windows: {len(windows)}")
    return windows
