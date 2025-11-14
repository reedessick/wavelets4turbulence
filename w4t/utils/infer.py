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

DEFAULT_NUM_RETAINED = np.inf

#-------------------------------------------------

def scaling_exponent_ansatz(index, x, C0, beta):
    """She-Leveque formula for the scaling exponents of structure function
    xi = (index/3)*(1-x) + C0*(1-beta**(index/3))
    based on Eq 7.64 of "Magnetohydrodynamic Turbulence" (Dieter Biskamp)
    """
    return (index/3)*(1-x) + C0*(1-beta**(index/3))

#------------------------

def _sample_sea_prior(
        indexes,
        ref_scale,
        min_x=0,
        max_x=1,
        min_C0=0,
        max_C0=3,
        min_beta=0,
        max_beta=1,
        mean_logsl=np.log(10),
        stdv_logsl=1.0,
        mean_bl=0.0,
        stdv_bl=3.0,
        mean_nl=0.0, ### FIXME may want to force this to be a roll-off --> nl, bl > 0 ?
        stdv_nl=3.0,
        mean_logsh=np.log(128),
        stdv_logsh=1.0,
        mean_bh=0.0,
        stdv_bh=3.0,
        mean_nh=0.0,
        stdv_nh=3.0,
        **ignored
    ):

    # sample the She-Leveque parametrization
    x, C0, beta = _sample_sea_xcb_prior(
        min_x=min_x,
        max_x=max_x,
        min_C0=min_C0,
        max_C0=max_C0,
        min_beta=min_beta,
        max_beta=max_beta,
    )

    # compute the predicted logarithmic derivative
    dlSdls = numpyro.deterministic('dlogSdlogs', scaling_exponent_ansatz(indexes, x, C0, beta)) 

    # set up a plate for all indexes
    with numpyro.plate('sfa', len(indexes)) as ind:
        # sample for all but 1 of the parameters of the structure function ansatz
        amp = _sample_sfa_amp_prior(mean_logamp=mean_logamp, stdv_logamp=stdv_logamp)

        sl, bl, nl = _sample_sfa_sbn_prior(
            mean_logs=mean_logsl,
            stdv_logs=stdv_logsl,
            mean_b=mean_bl,
            stdv_b=stdv_bl,
            mean_n=mean_nl,
            stdv_n=stdv_nl,
            suffix='l',
        )

        sh, bh, nh = _sample_sfa_sbn_prior(  
            mean_logs=mean_logsh,
            stdv_logs=stdv_logsh,
            mean_b=mean_bh,
            stdv_b=stdv_bh,
            mean_n=mean_nh,  
            stdv_n=stdv_nh,
            suffix='h',
        )

        # solve for the remaining parameter to match dlogSdlogs
        xi = numpyro.deterministic(
            'xi',
            dlSdls - logarithmic_derivative_ansatz(ref_scale, amp, 0.0, sl, bl, nl, sh, bh, nh),
        )

    # return
    return x, C0, beta, dlSdls, amp, xi, sl, bl, nl, sh, bh, nh

def _sample_sea_xcb_prior(
        min_x=0,
        max_x=1,
        min_C0=0,
        max_C0=3,
        min_beta=0,
        max_beta=1,
    ):
    x = numpyro.sample("x", dist.Uniform(min_x, max_x))
    C0 = numpyro.sample("C0", dist.Uniform(min_C0, max_C0))
    beta = numpyro.sample("beta", dist.Uniform(min_beta, max_beta))
    return x, C0, beta

#-----------

def sample_scaling_exponent_ansatz(
        scales,
        mom,
        std,
        indexes,
        ref_scale,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        num_retained=DEFAULT_NUM_RETAINED,
        seed=DEFAULT_SEED,
        verbose=False,
        **prior_kwargs
    ):
    """sample for the distribution of scaling exponents
    """
    if numpyro is None:
        raise ImportError('could not import numpyro')

    if verbose:
        print('defining model')

    def sample_posterior(obs):
        # draw from prior
        x, C0, beta, dlSdls, amp, xi, sl, bl, nl, sh, bh, nh = _sample_sea_prior(indexes, ref_scale, **prior_kwargs)

        with numpyro.plate('sfa_data', num_indexes):
            raise NotImplementedError('''
            # compute expected value
            sf = structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh)

            # compare to observed data
            numpyro.sample('mom', dist.Normal(sf, std), obs=obs)
            ''')

    #---

    # run the sampler

    if verbose:
        print('running sampler for prior with seed=%d for %d warmup and %d samples' % (seed, num_warmup, num_samples))

    mcmc = MCMC(NUTS(_sample_sea_prior), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(seed), indexes, ref_scale, **prior_kwargs)
    prior = mcmc.get_samples()

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)
    if num_retained < np.inf:
        if verbose:
            print('retaining the final %d samples' % num_retained)
        prior = dict((key, val[-num_retained:]) for key, val in prior.items())

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

    if num_retained < np.inf:
        if verbose:
            print('retaining the final %d samples' % num_retained)
        posterior = dict((key, val[-num_retained:]) for key, val in posterior.items())

    #---

    # return
    return posterior, prior

#-------------------------------------------------

def structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh):
    """
    SF(s, p) = <d_{x,s}^p>_x = amp * s**xi * (1 + (sl/s)**nl)**(bl/nl) * (1 + (s/sh)**nh)**(bh/nh)
    """
    return amp * scales**xi * (1 + (sl/scales)**nl)**(bl/nl) * (1 + (scales/sh)**nh)**(bh/nh)

#---

def logarithmic_derivative_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh):
    """
    logarithmic derivative of structure_function_ansatz with respect to scales
    """
    return xi - bl/(1 + (sl/scales)**-nl) + bh/(1 + (scales/sh)**-nh)

#------------------------

def _sample_sfa_prior(
        mean_logamp=-10.0,
        stdv_logamp=10.0,
        mean_xi=0.0,
        stdv_xi=3.0,
        mean_logsl=np.log(10),
        stdv_logsl=1.0,
        mean_bl=0.0,
        stdv_bl=3.0,
        mean_nl=0.0, ### FIXME may want to force this to be a roll-off --> nl, bl > 0 ?
        stdv_nl=3.0,
        mean_logsh=np.log(128),
        stdv_logsh=1.0,
        mean_bh=0.0,
        stdv_bh=3.0,
        mean_nh=0.0,
        stdv_nh=3.0,
        **ignored
    ):
    amp = _sample_sfa_amp_prior(mean_logamp=mean_logamp, stdv_logamp=stdv_logamp)

    xi = _sample_sfa_xi_prior(mean_xi=mean_xi, stdv_xi=stdv_xi)

    sl, bl, nl = _sample_sfa_sbn_prior(
        mean_logs=mean_logsl,
        stdv_logs=stdv_logsl,
        mean_b=mean_bl,
        stdv_b=stdv_bl,
        mean_n=mean_nl,
        stdv_n=stdv_nl,
        suffix='l',
    )

    sh, bh, nh = _sample_sfa_sbn_prior( 
        mean_logs=mean_logsh,
        stdv_logs=stdv_logsh,
        mean_b=mean_bh,
        stdv_b=stdv_bh,
        mean_n=mean_nh, 
        stdv_n=stdv_nh,
        suffix='h',
    )

    return amp, xi, sl, bl, nl, sh, bh, nh

def _sample_sfa_amp_prior(mean_logamp=-10.0, stdv_logamp=10.0):
    return numpyro.sample("amp", dist.LogNormal(mean_logamp, stdv_logamp))

def _sample_sfa_xi_prior(mean_xi=0.0, stdv_xi=3.0):
    return numpyro.sample("xi", dist.Normal(mean_xi, stdv_xi))

def _sample_sfa_sbn_prior(
        mean_logs=np.log(10),
        stdv_logs=1.0,
        mean_b=0.0,
        stdv_b=3.0,
        mean_n=0.0,
        stdv_n=3.0,
        suffix='',
    ):
    s = numpyro.sample("s"+suffix, dist.LogNormal(mean_logs, stdv_logs))
    b = numpyro.sample("b"+suffix, dist.Normal(mean_b, stdv_b))
    n = numpyro.sample("n"+suffix, dist.Normal(mean_n, stdv_n))
    return s, b, n

#---

def sample_structure_function_ansatz(
        scales,
        mom,
        std,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        num_retained=DEFAULT_NUM_RETAINED,
        seed=DEFAULT_SEED,
        verbose=False,
        **prior_kwargs
    ):
    """sample for parameters of a simple model for structure function scaling
    """
    if numpyro is None:
        raise ImportError('could not import numpyro')

    if verbose:
        print('defining model')

    def sample_posterior(obs):
        # draw from prior
        amp, xi, sl, bl, nl, sh, bh, nh = _sample_sfa_prior(**prior_kwargs)

        # compute expected value
        sf = structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh)

        # compare to observed data
        numpyro.sample('mom', dist.Normal(sf, std), obs=obs)

    #---

    # run the sampler

    if verbose:
        print('running sampler for prior with seed=%d for %d warmup and %d samples' % (seed, num_warmup, num_samples))

    mcmc = MCMC(NUTS(_sample_sfa_prior), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(seed), **prior_kwargs)
    prior = mcmc.get_samples()

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    if num_retained < np.inf:
        if verbose:
            print('retaining the final %d samples' % num_retained)
        prior = dict((key, val[-num_retained:]) for key, val in prior.items())

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

    if num_retained < np.inf:
        if verbose:
            print('retaining the final %d samples' % num_retained)
        posterior = dict((key, val[-num_retained:]) for key, val in posterior.items())

    #---

    # return
    return posterior, prior
