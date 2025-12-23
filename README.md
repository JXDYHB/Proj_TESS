# Exoplanet Transit Modeling Workflow

This project involves extracting light curves from TESS data, selecting potential transit windows, and performing joint transit MCMC modeling. The workflow consists of the following key steps:

## 1. Download TESS Light Curves (`download.py`)

The `download.py` script is used to download TESS light curves for a specific TIC ID.

- **Function**: Downloads TESS light curves for a given TIC ID and exposure time using the `lightkurve` library.
- **Input**: TIC ID, exposure time.
- **Output**: A dictionary containing the light curves for different sectors, indexed by `(sector, exptime)`.

## 2. Transit Detection and Window Selection (`transit_window.py`)

The `transit_window.py` script processes the downloaded light curves and detects potential transit events.

- **Function**: Identifies dips in the light curve that may correspond to transits. The center of the transit (`t0`) is refined by evaluating symmetry and other characteristics of the dip.
- **Transit Identification Logic**:
    - The function looks for significant dips in the flux, where the flux decreases below a certain threshold.
    - The symmetry around the dip (before and after the minimum) is evaluated to confirm the shape matches a typical transit.
    - The `t0` (transit center) is refined by minimizing the symmetry loss between the left and right sides of the dip.
    - It calculates the difference between the median flux at the center and at the outer regions, which helps in identifying whether the dip is sufficiently flat to be considered transit-like.
- **Input**: Time, flux, initial guess for `t0`, window size, and various parameters for dip detection and symmetry refinement.
- **Output**: A list of transit windows, each with the refined `t0`, symmetry loss, and whether the dip is transit-like.


## 3. Transit Model Definition and Fitting (`model_build.py`)

The `model_build.py` script defines the transit model and handles the MCMC sampling.

- **Function**: Defines a joint transit model that accounts for limb darkening, the transit light curve, and noise in the data. It then performs MCMC to estimate the model parameters.
- **Model Components**:
    - **Transit Model**: The model assumes a standard transit shape, with parameters including the transit duration, radius ratio (`r`), impact parameter (`b`), and baseline flux. Limb darkening is modeled using two parameters (`q1`, `q2`).
    - **Noise Model**: The model accounts for jitter in the measurements by adding a jitter term to the flux errors.
    - **MCMC Sampling**: The model is fitted using MCMC with `numpyro` and `NUTS` (No-U-Turn Sampler), which samples from the posterior distribution of the model parameters.
- **Input**: Data for each transit window (time, flux, error), and MCMC settings (e.g., number of chains, warmup steps).
- **Output**: Posterior samples, summary statistics (mean and standard deviation of the parameters), and model diagnostics.


## 4. MCMC Fitting (`union.py`)

The `union.py` script coordinates the entire process by combining data downloading, window selection, and MCMC fitting.

- **Input**: TESS sector data, sectors to process, window size, and MCMC parameters (e.g., number of warmup steps, number of samples, target acceptance probability).
- **Output**: MCMC results, including posterior samples for model parameters and summary statistics.
