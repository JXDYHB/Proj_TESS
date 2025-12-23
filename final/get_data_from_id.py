import lightkurve as lk

def download_all_sectors(tic_id, author="SPOC"):
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author=author)
    data = {}
    
    for row in search:
        # 有些 LightKurve 版本可以直接 row.sector；保险一点：
        mission_str = row.table["mission"][0]
        # "TESS Sector 92" -> 92
        sector_num = int(str(mission_str).split()[-1])
        key = (sector_num, row.table["exptime"][0])


        print(f"Downloading TIC {tic_id} Sector {sector_num} ...")
        lc = (row
              .download(flux_column="pdcsap_flux")
              .remove_nans()
              .normalize())

        data[key] = lc

    return data
