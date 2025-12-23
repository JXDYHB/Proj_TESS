# plot.py
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import arviz as az

from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve


def _kipping_q_to_u(q1, q2):
    s = jnp.sqrt(q1)
    return 2 * s * q2, s * (1 - 2 * q2)


def _model_flux(t, P, duration, r, b, q1, q2, c0=1.0, dt0=0.0):
    u1, u2 = _kipping_q_to_u(q1, q2)
    u = jnp.array([u1, u2])
    orbit = TransitOrbit(
        period=P,
        duration=duration,
        time_transit=dt0,
        impact_param=b,
        radius_ratio=r,
    )
    delta = limb_dark_light_curve(orbit, u)(t)
    return c0 * (1.0 + delta)


def plot_results(
    sector_data=None,
    sectors=None,
    tic=120,
    windows=None,
    idata=None,
    P_fixed=None,
    stack_step=0.01,
    show_sectors=False,
    show_windows=True,
    show_fit=True,
    show_corner=False,
    var_names=("r", "b", "logD", "q1", "q2", "sigma_jit"),
    point_ms=2,
):
    """
    Plot helpers for the pipeline:
      - sector light curves with selected transit centers
      - selected windows
      - model vs data per window (requires idata + P_fixed)
      - posterior corner/pair plot (requires idata)
    """
    windows = windows or []

    if show_sectors and (sector_data is not None) and (sectors is not None):
        centers_by_sector = {}
        for w in windows:
            if "sector" in w and "t_center" in w:
                centers_by_sector.setdefault(int(w["sector"]), []).append(float(w["t_center"]))

        for sec in sectors:
            lc = sector_data.get((sec, tic), None)
            if lc is None:
                continue
            time = np.asarray(lc.time.value, float)
            flux = np.asarray(lc.flux.value, float)
            m = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[m], flux[m]

            plt.figure(figsize=(12, 3))
            plt.plot(time, flux, "k.", ms=1, alpha=0.5)

            for t0 in centers_by_sector.get(int(sec), []):
                plt.axvline(t0, ls="--", lw=1)

            plt.title(f"Sector {sec} (selected centers marked)")
            plt.xlabel("Time [BTJD]")
            plt.ylabel("Flux")
            plt.tight_layout()
            plt.show()

    if show_windows and len(windows) > 0:
        plt.figure(figsize=(10, 4))
        for k, w in enumerate(windows):
            plt.plot(w["t"], w["f"] - stack_step * k, ".k", ms=point_ms)
        plt.title("Selected transit windows (stacked)")
        plt.xlabel("t - t_center [days]")
        plt.ylabel("Flux (offset)")
        plt.tight_layout()
        plt.show()

    if show_fit:
        if (idata is None) or (P_fixed is None) or (len(windows) == 0):
            pass
        else:
            post = idata.posterior

            r = float(post["r"].mean().values)
            b = float(post["b"].mean().values)
            duration = float(np.exp(post["logD"].mean().values))
            q1 = float(post["q1"].mean().values)
            q2 = float(post["q2"].mean().values)

            for k, w in enumerate(windows):
                t = np.asarray(w["t"], float)
                f = np.asarray(w["f"], float)

                dt_name = f"dt0_{k}"
                c0_name = f"c0_{k}"

                dt0 = float(post[dt_name].mean().values) if dt_name in post else 0.0
                c0 = float(post[c0_name].mean().values) if c0_name in post else 1.0

                t_grid = np.linspace(t.min(), t.max(), 400)
                f_model = np.asarray(_model_flux(jnp.asarray(t_grid), P_fixed, duration, r, b, q1, q2, c0=c0, dt0=dt0))

                plt.figure(figsize=(6, 3))
                plt.plot(t, f, "k.", ms=point_ms, label="data")
                plt.plot(t_grid, f_model, "-", lw=2, label="model (posterior mean)")
                plt.title(f"Window {k} (sector {w.get('sector', '?')})")
                plt.xlabel("t - t_center [days]")
                plt.ylabel("Normalized flux")
                plt.legend()
                plt.tight_layout()
                plt.show()

    if show_corner and (idata is not None):
        az.plot_pair(idata, var_names=list(var_names), kind="kde", marginals=True)
        plt.tight_layout()
        plt.show()
