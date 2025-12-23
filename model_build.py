# model_build.py
import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

import arviz as az

from transit_window import collect_windows


def kipping_q_to_u(q1, q2):
    s = jnp.sqrt(q1)
    return 2 * s * q2, s * (1 - 2 * q2)


def build_model_mcmc(
    sectors,
    sector_data,
    tic=120.0,
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
    Build windows and run joint transit MCMC across windows.
    Returns: windows, mcmc, samples, idata, summary
    """
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    numpyro.set_platform(platform)
    print("JAX devices:", jax.devices())

    windows = collect_windows(
        sectors=sectors,
        sector_data=sector_data,
        tic=float(tic),
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
    )

    if len(windows) == 0:
        raise RuntimeError("No accepted windows.")

    def joint_transit_model(windows_local):
        logD = numpyro.sample("logD", dist.Uniform(jnp.log(0.05), jnp.log(0.3)))
        duration = jnp.exp(logD)

        r = numpyro.sample("r", dist.Uniform(0.01, 0.2))
        b = numpyro.sample("b", dist.Uniform(0.0, 1.0))

        q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
        q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
        u1, u2 = kipping_q_to_u(q1, q2)
        u = jnp.array([u1, u2])

        sigma_jit = numpyro.sample("sigma_jit", dist.HalfNormal(5e-4))

        for k, w in enumerate(windows_local):
            t = jnp.asarray(w["t"])
            f = jnp.asarray(w["f"])
            e = jnp.asarray(w["e"])

            dt0 = numpyro.sample(f"dt0_{k}", dist.Normal(0.0, 0.05))
            c0 = numpyro.sample(f"c0_{k}", dist.Normal(1.0, 0.02))

            orbit = TransitOrbit(
                period=float(P_fixed),
                duration=duration,
                time_transit=dt0,
                impact_param=b,
                radius_ratio=r,
            )

            delta = limb_dark_light_curve(orbit, u)(t)
            model = c0 * (1.0 + delta)

            sigma_tot = jnp.sqrt(e**2 + sigma_jit**2)
            numpyro.sample(f"obs_{k}", dist.Normal(model, sigma_tot), obs=f)

    kernel = NUTS(
        joint_transit_model,
        target_accept_prob=float(target_accept_prob),
        max_tree_depth=int(max_tree_depth),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        progress_bar=True,
    )

    mcmc.run(jax.random.PRNGKey(int(rng_seed)), windows)
    mcmc.print_summary()

    samples = mcmc.get_samples()
    idata = az.from_numpyro(mcmc)

    sj = np.asarray(samples["sigma_jit"])
    summary = {
        "sigma_jit_mean": float(sj.mean()),
        "sigma_jit_std": float(sj.std()),
        "n_windows": int(len(windows)),
    }

    return windows, mcmc, samples, idata, summary
