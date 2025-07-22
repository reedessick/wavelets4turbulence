"""a module to run simple inference tasks
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#------------------------

try:
    import jax
    from jax import random
    from jax import numpy as jnp

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import (MCMC, NUTS)

    numpyro.enable_x64() # improve default numerical precision

except ImportError:
    numpyro = None

#-------------------------------------------------

DEFAULT_SEED = 1

DEFAULT_NUM_WARMUP = 500
DEFAULT_NUM_SAMPLES = 1000

#-------------------------------------------------

def structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh):
    """
    SF(s, p) = <d_{x,s}^p>_x = amp * s**xi * (1 + (sl/s)**nl)**(bl/nl) * (1 + (s/sh)**nh)**(bh/nh)
    """
    return amp * scales**xi * (1 + (sl/scales)**nl)**(bl/nl) * (1 + (scales/sh)**nh)**(bh/nh)

#------------------------

def sample_prior(
        mean_logamp=-10.0,
        stdv_logamp=10.0,
        mean_xi=0.0,
        stdv_xi=3.0,
        mean_logsl=np.log(10), ## FIXME
        stdv_logsl=2.0,
        mean_bl=0.0,
        stdv_bl=3.0,
        mean_nl=0.0,
        stdv_nl=3.0,
        mean_logsh=np.log(128),
        stdv_logsh=2.0,
        mean_bh=0.0,
        stdv_bh=3.0,
        mean_nh=0.0,
        stdv_nh=3.0,
    ):
    amp = numpyro.sample("amp", dist.LogNormal(mean_logamp, stdv_logamp))
    xi = numpyro.sample("xi", dist.Normal(mean_xi, stdv_xi))

    sl = numpyro.sample("sl", dist.LogNormal(mean_logsl, stdv_logsl))
    bl = numpyro.sample("bl", dist.Normal(mean_bl, stdv_bl))
    nl = numpyro.sample("nl", dist.Normal(mean_nl, stdv_nl))

    sh = numpyro.sample("sh", dist.LogNormal(mean_logsh, stdv_logsh))
    bh = numpyro.sample("bh", dist.Normal(mean_bh, stdv_bh))
    nh = numpyro.sample("nh", dist.Normal(mean_nh, stdv_nh))

    return amp, xi, sl, bl, nl, sh, bh, nh

#---

def sample_structure_function_ansatz(
        scales,
        mom,
        std,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        seed=DEFAULT_SEED,
        verbose=False,
        **prior_kwargs
    ):
    """sample for parameters of a simple model for structure function scaling
    """
    if numpyro is None:
        raise NotImplementedError('could not import numpyro!')

    if verbose:
        print('defining model')

    def sample_posterior(obs):
        # draw from prior
        amp, xi, sl, bl, nl, sh, bh, nh = sample_prior(**prior_kwargs)

        # compute expected value
        sf = structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh)

        # compare to observed data
        numpyro.sample('mom', dist.Normal(sf, std), obs=obs)

    #---

    # run the sampler

    if verbose:
        print('running sampler for prior with seed=%d for %d warmup and %d samples' % (seed, num_warmup, num_samples))

    mcmc = MCMC(NUTS(sample_prior), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(seed))
    prior = mcmc.get_samples()

#    # record the likelihood of each sample
#
#    if verbose:
#        print('computing likelihood at samples')
#
#    prior.update(numpyro.infer.log_likelihood(sample_prior, prior, None))

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    #---

    if verbose:
        print('running sampler for posterior with seed=%d for %d warmup and %d samples' % (seed, num_warmup, num_samples))

    mcmc = MCMC(NUTS(sample_posterior), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(seed), mom)
    posterior = mcmc.get_samples()

    # record the likelihood of each sample

    if verbose:
        print('computing likelihood at samples')

    posterior.update(numpyro.infer.log_likelihood(sample_posterior, posterior, mom))

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    #---

    # retur
    return posterior, prior
