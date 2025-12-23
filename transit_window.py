# transit_window.py
import numpy as np
from scipy.signal import find_peaks


def _mad_sigma(x):
    """Robust scale (Gaussian-equivalent sigma) via MAD; fallback to std."""
    x = np.asarray(x, float)
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 0:
        s = np.std(x)
    return float(s)


def coarse_dip_center(t_win, f_win, n_low=15):
    """Coarse center = weighted mean time of the lowest n_low normalized flux points."""
    t = np.asarray(t_win, float)
    f = np.asarray(f_win, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 30:
        return np.nan

    base = np.median(f)
    if not np.isfinite(base) or base == 0:
        return np.nan
    fn = f / base

    k = min(int(n_low), fn.size)
    idx = np.argsort(fn)[:k]
    t_low = t[idx]
    w = 1.0 / (np.arange(1, k + 1))
    return float(np.sum(w * t_low) / np.sum(w))


def symmetry_loss_at_t0(t, f_norm, t0, tol, min_pairs=12):
    """Robust symmetry loss around t0 using matched left-right pairs within tol."""
    t = np.asarray(t, float)
    f_norm = np.asarray(f_norm, float)

    tau = t - float(t0)
    tp = tau[tau > 0]
    fp = f_norm[tau > 0]
    tn = -tau[tau < 0]
    fn = f_norm[tau < 0]
    if tp.size < 8 or tn.size < 8:
        return np.inf, 0

    ip = np.argsort(tp)
    tp, fp = tp[ip], fp[ip]
    in_ = np.argsort(tn)
    tn, fn = tn[in_], fn[in_]

    pairs_r, pairs_l = [], []
    for ti, fi in zip(tp, fp):
        j = np.searchsorted(tn, ti, side="left")
        cand = []
        if 0 <= j - 1 < tn.size:
            cand.append(j - 1)
        if 0 <= j < tn.size:
            cand.append(j)
        if 0 <= j + 1 < tn.size:
            cand.append(j + 1)
        if not cand:
            continue

        jj = min(cand, key=lambda k: abs(tn[k] - ti))
        if abs(tn[jj] - ti) <= tol:
            pairs_r.append(fi)
            pairs_l.append(fn[jj])

    pairs_r = np.asarray(pairs_r, float)
    pairs_l = np.asarray(pairs_l, float)
    n_pairs = int(pairs_r.size)
    if n_pairs < int(min_pairs):
        return np.inf, n_pairs

    resid = pairs_r - pairs_l
    s = _mad_sigma(resid)
    if not np.isfinite(s) or s <= 1e-10:
        return np.inf, n_pairs

    loss = float(np.mean(np.abs(resid) / s))
    return loss, n_pairs


def refine_center_by_symmetry(
    t_win,
    f_win,
    t0_coarse,
    refine_halfspan_cadences=3,
    n_scan=21,
    min_pairs=12,
):
    """Scan around t0_coarse and choose t0 minimizing symmetry loss."""
    t = np.asarray(t_win, float)
    f = np.asarray(f_win, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 30:
        return np.nan, np.inf, 0

    base = np.median(f)
    if not np.isfinite(base) or base == 0:
        return np.nan, np.inf, 0
    fn = f / base

    ts = np.sort(t)
    dt = np.median(np.diff(ts))
    if not np.isfinite(dt) or dt <= 0:
        dt = (t.max() - t.min()) / 200.0

    tol = 0.75 * dt
    halfspan = float(refine_halfspan_cadences) * dt
    grid = np.linspace(float(t0_coarse) - halfspan, float(t0_coarse) + halfspan, int(n_scan))

    best_t0, best_loss, best_pairs = np.nan, np.inf, 0
    for t0 in grid:
        loss, npairs = symmetry_loss_at_t0(t, fn, float(t0), tol=tol, min_pairs=min_pairs)
        if loss < best_loss:
            best_loss, best_t0, best_pairs = float(loss), float(t0), int(npairs)

    return best_t0, best_loss, best_pairs


def edge_rise_metrics(t_win, f_win, t0, window, center_frac=0.25, edge_frac=0.70):
    """Edges should be higher than the center for a transit-like dip."""
    t = np.asarray(t_win, float)
    f = np.asarray(f_win, float)

    base = np.median(f)
    if not np.isfinite(base) or base == 0:
        return np.nan, np.nan, np.nan

    fn = f / base
    trel = t - float(t0)

    center_mask = np.abs(trel) <= float(center_frac) * float(window)
    edge_mask = np.abs(trel) >= float(edge_frac) * float(window)

    if np.sum(center_mask) < 5 or np.sum(edge_mask) < 8:
        return np.nan, np.nan, np.nan

    f_c = fn[center_mask]
    f_e = fn[edge_mask]

    med_c = np.median(f_c)
    med_e = np.median(f_e)

    sigma_e = _mad_sigma(f_e)
    rise = float(med_e - med_c)
    snr_rise = float(rise / sigma_e) if np.isfinite(sigma_e) and sigma_e > 0 else np.nan
    return snr_rise, rise, float(sigma_e)


def plateau_centerK_vs_rest(t_win, f_win, t0, K=5):
    """Center K points flatness compared to the rest."""
    t = np.asarray(t_win, float)
    f = np.asarray(f_win, float)

    base = np.median(f)
    if not np.isfinite(base) or base == 0:
        return np.nan, np.nan, np.nan, np.nan

    fn = f / base
    trel = t - float(t0)

    order = np.argsort(np.abs(trel))
    kk = min(int(K), order.size)
    if kk < 3 or (order.size - kk) < 10:
        return np.nan, np.nan, np.nan, np.nan

    idx_c = order[:kk]
    idx_r = order[kk:]
    f_c = fn[idx_c]
    f_r = fn[idx_r]

    med_c = np.median(f_c)
    med_r = np.median(f_r)

    sigma_r = _mad_sigma(f_r)
    scat_c = _mad_sigma(f_c)

    delta = float(med_r - med_c)
    snr_loc = float(delta / sigma_r) if np.isfinite(sigma_r) and sigma_r > 0 else np.nan
    flat_ratio = float(scat_c / sigma_r) if np.isfinite(sigma_r) and sigma_r > 0 else np.nan
    return snr_loc, flat_ratio, delta, float(sigma_r)


def score_window_final_v3(
    time,
    flux,
    t_guess,
    window=0.5,
    # centering
    n_low=15,
    refine_halfspan_cadences=3,
    n_scan=21,
    min_pairs=12,
    # plateau/edge-rise
    K=5,
    flat_thr=1.5,
    snr_rise_thr=1.5,
    center_frac=0.25,
    edge_frac=0.70,
    # dip threshold
    dip_sigma=3.0,
    inner_frac=0.5,
    # count-based dip requirement
    min_dip_points=6,
    min_span_cadences=3,
):
    """
    v3:
      - cut window around t_guess
      - coarse dip center -> symmetry-refined t0_sym
      - accept if edge-rise + flatness pass
      - extra guard: within inner region, require enough points below threshold AND non-trivial time span
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)

    m = np.abs(time - float(t_guess)) < float(window)
    t_win = time[m]
    f_win = flux[m]
    if t_win.size < 30:
        return dict(ok=False, reason="too few points")

    t0_coarse = coarse_dip_center(t_win, f_win, n_low=n_low)
    if not np.isfinite(t0_coarse):
        return dict(ok=False, reason="coarse failed")

    t0_sym, sym_loss, sym_pairs = refine_center_by_symmetry(
        t_win, f_win, t0_coarse,
        refine_halfspan_cadences=refine_halfspan_cadences,
        n_scan=n_scan,
        min_pairs=min_pairs,
    )
    if not np.isfinite(t0_sym):
        return dict(ok=False, reason="sym refine failed")

    snr_loc, flat_ratio, _, _ = plateau_centerK_vs_rest(t_win, f_win, t0_sym, K=K)
    if not np.isfinite(flat_ratio):
        return dict(ok=False, reason="plateau failed")

    snr_rise, rise, _ = edge_rise_metrics(
        t_win, f_win, t0_sym, window=window, center_frac=center_frac, edge_frac=edge_frac
    )
    if not np.isfinite(snr_rise):
        return dict(ok=False, reason="edge-rise failed")

    # normalize on the window
    base = np.median(f_win)
    if not np.isfinite(base) or base == 0:
        return dict(ok=False, reason="baseline failed")
    fn = f_win / base

    trel = t_win - float(t0_sym)
    abs_t = np.abs(trel)

    inner_mask = abs_t <= float(inner_frac) * float(window)
    if np.sum(inner_mask) < 10:
        return dict(ok=False, reason="inner too few points")

    edge_mask = abs_t >= float(edge_frac) * float(window)
    n_edge = int(np.sum(edge_mask))
    base_edge = float(np.median(fn[edge_mask])) if n_edge >= 8 else np.nan
    sigma_dip = _mad_sigma(fn[edge_mask]) if n_edge >= 8 else _mad_sigma(fn)
    if not np.isfinite(sigma_dip) or sigma_dip <= 0:
        return dict(ok=False, reason="sigma failed")

    thresh = (base_edge - float(dip_sigma) * sigma_dip) if n_edge >= 8 else (1.0 - float(dip_sigma) * sigma_dip)
    dip_bool = (fn < thresh) & inner_mask

    # count + span
    ts = np.sort(t_win)
    dt = np.median(np.diff(ts))
    if not np.isfinite(dt) or dt <= 0:
        dt = (t_win.max() - t_win.min()) / 200.0

    dip_idx = np.where(dip_bool)[0]
    n_dip = int(dip_idx.size)
    dip_span = float(t_win[dip_idx].max() - t_win[dip_idx].min()) if n_dip > 0 else 0.0

    dbg = dict(
        window=float(window),
        inner_frac=float(inner_frac),
        edge_frac=float(edge_frac),
        dip_sigma=float(dip_sigma),
        min_dip_points=int(min_dip_points),
        min_span_cadences=int(min_span_cadences),
        n_edge=int(n_edge),
        n_inner=int(np.sum(inner_mask)),
        base_edge=float(base_edge) if np.isfinite(base_edge) else np.nan,
        sigma_dip=float(sigma_dip),
        thresh=float(thresh),
        n_dip=int(n_dip),
        dip_span=float(dip_span),
        dt=float(dt),
        t0_sym=float(t0_sym),
    )

    if (n_dip < int(min_dip_points)) or (dip_span < float(min_span_cadences) * float(dt)):
        return dict(ok=False, reason="dip-count failed", dbg=dbg)

    is_transit_like = (snr_rise >= float(snr_rise_thr)) and (flat_ratio <= float(flat_thr))

    return dict(
        ok=True,
        is_transit_like=bool(is_transit_like),
        t_guess=float(t_guess),
        t0_sym=float(t0_sym),
        sym_loss=float(sym_loss),
        sym_pairs=int(sym_pairs),
        snr_rise=float(snr_rise),
        rise=float(rise),
        flat_ratio=float(flat_ratio),
        snr_loc=float(snr_loc) if np.isfinite(snr_loc) else np.nan,
        n_dip=int(n_dip),
        dip_span=float(dip_span),
        sigma_dip=float(sigma_dip),
        dip_thresh=float(thresh),
        dbg=dbg,
    )


def cut_transit_window(
    time,
    flux,
    t_center,
    window=0.5,
    # centering/symmetry
    n_low=15,
    refine_halfspan_cadences=3,
    n_scan=21,
    min_pairs=12,
    # shape thresholds
    K=5,
    flat_thr=1.5,
    snr_rise_thr=1.5,
    center_frac=0.25,
    edge_frac=0.70,
    # dip threshold + count gate
    dip_sigma=3.0,
    inner_frac=0.5,
    min_dip_points=6,
    min_span_cadences=3,
):
    """Return MCMC-ready window dict if transit-like, else None."""
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)

    res = score_window_final_v3(
        time=time,
        flux=flux,
        t_guess=float(t_center),
        window=float(window),
        n_low=n_low,
        refine_halfspan_cadences=refine_halfspan_cadences,
        n_scan=n_scan,
        min_pairs=min_pairs,
        K=K,
        flat_thr=flat_thr,
        snr_rise_thr=snr_rise_thr,
        center_frac=center_frac,
        edge_frac=edge_frac,
        dip_sigma=dip_sigma,
        inner_frac=inner_frac,
        min_dip_points=min_dip_points,
        min_span_cadences=min_span_cadences,
    )
    if (not res.get("ok", False)) or (not res.get("is_transit_like", False)):
        return None

    t0 = float(res["t0_sym"])

    m2 = np.abs(time - t0) < float(window)
    t2 = time[m2]
    f2_raw = flux[m2]
    if t2.size < 30:
        return None

    baseline = np.median(f2_raw)
    if not np.isfinite(baseline) or baseline == 0:
        return None

    f2 = f2_raw / baseline
    t_rel = t2 - t0

    _, _, sigma_e = edge_rise_metrics(
        t2, f2_raw, t0, window=window, center_frac=center_frac, edge_frac=edge_frac
    )
    sigma = float(sigma_e) if np.isfinite(sigma_e) and sigma_e > 0 else _mad_sigma(f2)
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    e = np.ones_like(f2) * sigma

    return dict(
        t=t_rel,
        f=f2,
        e=e,
        t_center=float(t0),
        t_guess=float(t_center),
        snr_rise=float(res["snr_rise"]),
        rise=float(res["rise"]),
        flat_ratio=float(res["flat_ratio"]),
        sym_loss=float(res["sym_loss"]),
        sym_pairs=int(res["sym_pairs"]),
        snr_loc=float(res["snr_loc"]) if np.isfinite(res["snr_loc"]) else np.nan,
        n_dip=int(res["n_dip"]),
        dip_span=float(res["dip_span"]),
        dip_thresh=float(res["dip_thresh"]),
    )


def collect_windows(
    sectors,
    sector_data,
    tic=120.0,
    window=0.5,
    max_windows_per_sector=2,
    prom_sigma=3.0,
    distance_cap=1000,
    # scoring params
    n_low=15,
    refine_halfspan_cadences=3,
    n_scan=21,
    min_pairs=12,
    K=5,
    flat_thr=1.5,
    snr_rise_thr=1.5,
    center_frac=0.25,
    edge_frac=0.70,
    dip_sigma=3.0,
    inner_frac=0.5,
    min_dip_points=6,
    min_span_cadences=3,
):
    """Find prominent dips and keep up to max_windows_per_sector transit-like windows per sector."""
    windows = []

    for sector in sectors:
        print(f"\n--- Processing Sector {sector} ---")
        lc = sector_data.get((int(sector), float(tic)), None)
        if lc is None:
            print(f"  (skip) sector_data has no key {(int(sector), float(tic))}")
            continue

        time = np.asarray(lc.time.value, float)
        flux = np.asarray(lc.flux.value, float)

        good = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[good], flux[good]
        if time.size < 100:
            continue

        lo, hi = np.nanpercentile(flux, [0.1, 99.9])
        flux = np.clip(flux, lo, hi)

        inv_flux = 1.0 - flux
        sig = float(np.std(inv_flux))
        dist_min = min(int(distance_cap), int(len(inv_flux) // 2))

        peaks, props = find_peaks(inv_flux, prominence=float(prom_sigma) * sig, distance=dist_min)
        if len(peaks) == 0:
            continue

        prominences = np.asarray(props.get("prominences", np.zeros(len(peaks))), float)
        order = np.argsort(prominences)[::-1]

        accepted = 0
        for j in order:
            if accepted >= int(max_windows_per_sector):
                break

            idx = int(peaks[j])
            t_guess = float(time[idx])

            win = cut_transit_window(
                time=time,
                flux=flux,
                t_center=t_guess,
                window=window,
                n_low=n_low,
                refine_halfspan_cadences=refine_halfspan_cadences,
                n_scan=n_scan,
                min_pairs=min_pairs,
                K=K,
                flat_thr=flat_thr,
                snr_rise_thr=snr_rise_thr,
                center_frac=center_frac,
                edge_frac=edge_frac,
                dip_sigma=dip_sigma,
                inner_frac=inner_frac,
                min_dip_points=min_dip_points,
                min_span_cadences=min_span_cadences,
            )
            if win is None:
                continue

            win["sector"] = int(sector)
            win["prominence"] = float(prominences[j]) if prominences.size else np.nan
            windows.append(win)
            accepted += 1

    print(f"\nTotal accepted windows: {len(windows)}")
    return windows
