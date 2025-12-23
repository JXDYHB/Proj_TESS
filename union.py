# union.py
from download import download_sector
from model_build import build_model_mcmc


def run_union(
    tic_id,
    sectors,
    tic=120.0,
    author="SPOC",
    P_fixed=9.477,
    window=0.5,
    max_windows_per_sector=2,
    prom_sigma=3.0,
    distance_cap=1000,
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
    num_warmup=400,
    num_samples=2000,
    num_chains=1,
    target_accept_prob=0.92,
    max_tree_depth=9,
    rng_seed=0,
    platform="gpu",
    enable_x64=True,
):
    """
    Download -> window selection -> MCMC.
    Returns a dict usable by plot.plot_results(**out, ...).
    """
    sectors = [int(s) for s in sectors]

    data_all = download_sector(int(tic_id), author=author)

    sector_data = {}
    for (sec, exptime), lc in data_all.items():
        if int(sec) in sectors and float(exptime) == float(tic):
            sector_data[(int(sec), float(exptime))] = lc

    windows, mcmc, samples, idata, summary = build_model_mcmc(
        sectors=sectors,
        sector_data=sector_data,
        tic=float(tic),
        P_fixed=float(P_fixed),
        window=float(window),
        max_windows_per_sector=int(max_windows_per_sector),
        prom_sigma=float(prom_sigma),
        distance_cap=int(distance_cap),
        n_low=int(n_low),
        refine_halfspan_cadences=int(refine_halfspan_cadences),
        n_scan=int(n_scan),
        min_pairs=int(min_pairs),
        K=int(K),
        flat_thr=float(flat_thr),
        snr_rise_thr=float(snr_rise_thr),
        center_frac=float(center_frac),
        edge_frac=float(edge_frac),
        dip_sigma=float(dip_sigma),
        inner_frac=float(inner_frac),
        min_dip_points=int(min_dip_points),
        min_span_cadences=int(min_span_cadences),
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        target_accept_prob=float(target_accept_prob),
        max_tree_depth=int(max_tree_depth),
        rng_seed=int(rng_seed),
        platform=str(platform),
        enable_x64=bool(enable_x64),
    )

    return {
        "sector_data": sector_data,
        "sectors": sectors,
        "tic": float(tic),
        "windows": windows,
        "idata": idata,
        "P_fixed": float(P_fixed),
    }
