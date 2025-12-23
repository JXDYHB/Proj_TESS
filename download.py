# download.py
import lightkurve as lk
import numpy as np

def download_sector(tic_id, author="SPOC", target_exptime=120.0):
    """
    Download TESS PDCSAP light curves for a given TIC ID,
    restricted to a specific exposure time (default: 120s).
    """
    search = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        author=author
    )
    tbl = search.table
    data = {}

    for sec in np.unique(tbl["sequence_number"]):
        mask_sec = tbl["sequence_number"] == sec
        sub = tbl[mask_sec]

        mask_exp = np.isclose(sub["exptime"], target_exptime)
        if not np.any(mask_exp):
            continue

        row = search[mask_sec][mask_exp][0]
        lc = (
            row.download(flux_column="pdcsap_flux")
               .remove_nans()
               .normalize()
        )

        data[(int(sec), float(target_exptime))] = lc

    return data
