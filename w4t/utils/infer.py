"""a module to run simple inference tasks
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#------------------------

import jax
from jax import random
from jax import numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (MCMC, NUTS)

numpyro.enable_x64() # improve default numerical precision

#-------------------------------------------------

DEFAULT_SEED = 1

DEFAULT_NUM_WARMUP = 500
DEFAULT_NUM_SAMPLES = 1000

#-------------------------------------------------

def structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh):
    """
    SF(s, p) = <d_{x,s}^p>_x = amp * s**xi * (1 + (sl/s)**nl)**bl * (1 + (s/sh)**nh)**bh
    """
    return amp * scales**xi * (1 + (sl/scales)**nl)**bl * (1 + (scales/sh)**nh)**bh

#------------------------

def scaling_exponents(
        scales,
        mom,
        std,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        seed=DEFAULT_SEED,
        verbose=False,
    ):
    """sample for parameters of a simple model for structure function scaling
    """
    if verbose:
        print('defining model')

    s0_loc = np.mean(np.log(scales))
    s0_scale = np.log(np.max(scales)/np.min(scales)) * 0.5

    def model(obs):
        # draw from prior
        amp = numpyro.sample("amp", dist.LogNormal(-10, 5.0))
        xi = numpyro.sample("xi", dist.Normal(0.0, 3.0))

        sl = numpyro.sample("sl", dist.LogNormal(s0_loc, s0_scale))
        bl = numpyro.sample("bl", dist.Normal(0.0, 3.0))
        nl = numpyro.sample("nl", dist.Exponential(1.0))

        sh = numpyro.sample("sh", dist.LogNormal(s0_loc, s0_scale))
        bh = numpyro.sample("bh", dist.Normal(0.0, 3.0))
        nh = numpyro.sample("nh", dist.Exponential(1.0))

        # compute expected value
        sf = numpyro.deterministic('structure_function', structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh))

        numpyro.sample('mom', dist.Normal(sf, std), obs=obs)

    #---

    # instantiate the sampler
    if verbose:
        print('instantiating sampler')

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples)

    # run the sample

    if verbose:
        print('running sampler for prior with seed=%d for %d warmup and %d samples' % (seed, num_warmup, num_samples))

    mcmc.run(random.PRNGKey(seed), None)
    prior = mcmc.get_samples()

    # record the likelihood of each sample

    if verbose:
        print('computing likelihood at samples')

    prior.update(numpyro.infer.log_likelihood(model, prior, None))

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    #---

    if verbose:
        print('running sampler for posterior with seed=%d for %d warmup and %d samples' % (seed, num_warmup, num_samples))

    mcmc.run(random.PRNGKey(seed), mom)
    posterior = mcmc.get_samples()

    # record the likelihood of each sample

    if verbose:
        print('computing likelihood at samples')

    posterior.update(numpyro.infer.log_likelihood(model, posterior, mom))

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    #---

    # retur
    return posterior, prior
