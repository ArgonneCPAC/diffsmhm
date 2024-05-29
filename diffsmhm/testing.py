import numpy as np


def gen_mstar_data(
    *, seed, npts=100, npars=3, nds=10, boxsize=120, zmax=20, nbins=10,
    rpmax=25
):
    rng = np.random.RandomState(seed=seed)

    log10mstar = rng.uniform(size=npts)
    log10mstar_jac = rng.uniform(size=(npars, npts))
    sigma = rng.uniform(size=npts)
    sigma_jac = rng.uniform(size=(npars, npts))
    w = rng.uniform(size=npts)
    w_jac = rng.uniform(size=(npars, npts))
    ds_per_object = rng.uniform(size=(npts, nds))
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0])
    x = rng.uniform(size=npts) * boxsize
    y = rng.uniform(size=npts) * boxsize
    z = rng.uniform(size=npts) * boxsize
    rp_bins = np.logspace(-1, np.log10(rpmax), nbins + 1, dtype=np.float64)

    if rp_bins[0] > 0:
        np.concatenate([0.0], rp_bins)

    return dict(
        log10mstar=log10mstar,
        log10mstar_jac=log10mstar_jac,
        sigma=sigma,
        sigma_jac=sigma_jac,
        npars=npars,
        npts=npts,
        bins=bins,
        ds_per_object=ds_per_object,
        x=x,
        y=y,
        z=z,
        zmax=zmax,
        rp_bins=rp_bins,
        boxsize=boxsize,
        w=w,
        w_jac=w_jac,
    )
