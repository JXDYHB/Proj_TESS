# transit_window.py
import numpy as np
from scipy.signal import find_peaks


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
    Cut and vet a candidate transit window centered at t_center.

    The window is accepted only if it shows:
      - sufficient depth SNR,
      - multiple consecutive in-transit points,
      - approximate left-right symmetry.

    Returns a dict of windowed data if accepted, otherwise None.
    """
    # window cut
    mask = np.abs(time - t_center) < window
    t = np.asarray(time[mask], float)
    f_raw = np.asarray(flux[mask], float)
    if len(t) < 30:
        return None

    # normalization
    baseline = np.median(f_raw)
    if not np.isfinite(baseline) or baseline == 0:
        return None
    f = f_raw / baseline
    t = t - t_center

    # sort for consecutive-point logic
    sidx = np.argsort(t)
    t = t[sidx]
    f = f[sidx]

    # noise estimate from window edges
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

    # require multiple consecutive in-transit points near center
    inner_mask = abs_t < (inner_frac * window)
    if np.sum(inner_mask) < max(10, min_consecutive):
        return None

    base = np.median(f_edge)
    thresh = base - dip_sigma * sigma
    dip = (f < thresh) & inner_mask

    max_run = 0
    run = 0
    for v in dip:
        run = run + 1 if v else 0
        max_run = max(max_run, run)

    if max_run < min_consecutive:
        return None

    # depth SNR
    depth = base - np.min(f)
    snr = depth / sigma
    if snr < 5.0:
        return None

    # symmetry check
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
    Identify and extract transit-like windows from multiple sectors.

    Parameters
    ----------
    sector_data : dict
        Mapping (sector, tic) -> LightCurve.
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

        # clip extreme outliers
        lo, hi = np.nanpercentile(flux, [0.1, 99.9])
        flux = np.clip(flux, lo, hi)

        # detect candidate dips
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

        # test strongest dips first
        order = np.argsort(props["prominences"])[::-1]
        peaks = peaks[order][:max_windows_per_sector]

        for idx in peaks:
            win = cut_transit_window(time, flux, time[idx], window)
            if win is not None:
                win["sector"] = sector
                windows.append(win)

    print(f"\nTotal accepted windows: {len(windows)}")
    return windows
