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
    from numpyro.infer import (MCMC, NUTS, init_to_value)
    from numpyro.diagnostics import effective_sample_size

    numpyro.enable_x64() # improve default numerical precision

except ImportError:
    numpyro = None

#-------------------------------------------------

DEFAULT_SEED = 1

DEFAULT_NUM_WARMUP = 500
DEFAULT_NUM_SAMPLES = 1000

DEFAULT_NUM_RETAINED = np.inf

#-------------------------------------------------

def structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh):
    """
    SF(s, p) = <d_{x,s}^p>_x = amp * s**xi * (1 + (sl/s)**nl)**(bl/nl) * (1 + (s/sh)**nh)**(bh/nh)
    """
    return amp * scales**xi * (1 + (sl/scales)**nl)**(bl/nl) * (1 + (scales/sh)**nh)**(bh/nh)

_vmap_structure_function_ansatz = jax.vmap(
    structure_function_ansatz,
    in_axes=[0]+[None]*8,
)

#---

def logarithmic_derivative_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh):
    """
    logarithmic derivative of structure_function_ansatz with respect to scales
    """
    return xi - bl/(1 + (sl/scales)**-nl) + bh/(1 + (scales/sh)**-nh)

def averaged_logarithmic_derivative_ansatz(min_scale, max_scale, amp, xi, sl, bl, nl, sh, bh, nh):
    """computes the average of the logarithmic derivative between min_scale, max_scale
    """
    return xi + ((bl/nl)*jnp.log((1+(sl/max_scale)**nl)/(1+(sl/min_scale)**nl)) + (bh/nh)*jnp.log((1+(max_scale/sh)**nh)/(1+(min_scale/sh)**nh))) / jnp.log(max_scale/min_scale)

#---

def scaling_exponent_ansatz(index, x, C0, beta):
    """She-Leveque formula for the scaling exponents of structure function
    xi = (index/3)*(1-x) + C0*(1-beta**(index/3))
    based on Eq 7.64 of vi"Magnetohydrodynamic Turbulence" (Dieter Biskamp)
    """
    return (index/3)*(1-x) + C0*(1-beta**(index/3))

#-------------------------------------------------

def thin(num_samples, samples, keys, num_segs=1, verbose=False):
    """downsample based on estimate of effective sample size
    can downsample different parts of the chains differently, and will divide the chain into "num_segs" segments
    """
    samples = dict((k, np.array(v)) for k, v in samples.items()) # cast to numpy array

    # figure out boundaries for different segments
    if num_segs > 1:
        if verbose:
            print('splitting chains of length %d into %d segments' % (num_samples, num_segs))
        step = num_samples // num_segs
        segs = []
        start = 0
        stop = start+step
        while stop < num_samples:
            segs.append((start, stop))
            start = stop
            stop += step
        segs.append((start, num_samples))

    else:
        segs = [(0, num_samples)]

    if verbose:
        print('thinning separately in %d segments' % len(segs))
    ans = dict((k, []) for k in samples.keys())
    num = 0
    for start, end in segs:
        if verbose:
            print('    %d -> %d' % (start, end))

        ber, wer = _thin(end-start, dict((k, v[start:end]) for k, v in samples.items()), keys, verbose=verbose)
        num += ber
        for k in ans.keys():
            ans[k].append(wer[k])

    if verbose:
        print('retained a total of %d samples' % num)
    return dict((k, np.concatenate(tuple(v))) for k, v in ans.items())

#---

def _thin(num_samples, samples, keys=None, verbose=False):
    if keys is None:
        keys = list(samples.keys())

    min_neff = num_samples

    for k in keys:
        neff = effective_sample_size(samples[k].reshape((1,num_samples)))
        if verbose:
            print('    neff(%s) = %.3f' % (k, neff))
        min_neff = min(neff, min_neff)
    if verbose:
        print('    min neff = %.3f' % min_neff)

#    skip = int(np.ceil(num_samples/min_neff))
    skip = int(round(num_samples/min_neff, 0))

    num = num_samples/skip
    if verbose:
        print('    retaining 1 out of every %d steps --> %d samples' % (skip, num))
    # do this fancy indexing to prefer later samples over earlier samples
    return num, dict((k, v[::-skip][::-1]) for k, v in samples.items())

#-------------------------------------------------

def _sample_sea_prior(
        indexes,
        ref_scale,
        mean_logamp=-10.0,
        stdv_logamp=10.0,
        mean_logsl=np.log(10), # FIXME should be an array of the same length as indexes
        stdv_logsl=1.0,
        mean_bl=0.0,
        stdv_bl=3.0,
        mean_nl=0.0,
        stdv_nl=3.0,
        mean_logsh=np.log(128), # FIXME should be an array of the same length as indexes
        stdv_logsh=1.0,
        mean_bh=0.0,
        stdv_bh=3.0,
        mean_nh=0.0,
        stdv_nh=3.0,
        **xcb_prior_kwargs
    ):

    # sample the She-Leveque parametrization
    x, C0, beta = _sample_sea_xcb_prior(**xcb_prior_kwargs)

    # compute the predicted logarithmic derivative
    dlSdls = numpyro.deterministic('dlogSdlogs', scaling_exponent_ansatz(indexes, x, C0, beta)) 

    # set up a plate for all indexes
    with numpyro.plate('sfa', len(indexes)) as ind:
        # sample for all but 1 of the parameters of the structure function ansatz
        amp = _sample_sfa_amp_prior(mean_logamp=mean_logamp[ind], stdv_logamp=stdv_logamp)

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

        if isinstance(ref_scale, (int, float)): # a single reference scale
            xi = numpyro.deterministic(
                'xi',
                dlSdls - logarithmic_derivative_ansatz(ref_scale, amp, 0.0, sl, bl, nl, sh, bh, nh),
            )

        elif len(ref_scale) == 2: # defines a range over which we average
            xi = numpyro.deterministic(
                'xi',
                dlSdls - averaged_logarithmic_derivative_ansatz(*ref_scale, amp, 0.0, sl, bl, nl, sh, bh, nh),
            )

        else:
            raise ValueError('ref_scale=%s not understood!' % ref_scale)

    # return
    return x, C0, beta, dlSdls, amp, xi, sl, bl, nl, sh, bh, nh

#-----------

def _sample_sea_xcb_prior_uniform(
        min_C0=-1.0, # these are based on physics at least a bit
        max_C0=4.0,
        min_x=-1.0,  # these are wild guesses
        max_x=2.0,
        min_beta=0.0, # as are these
        max_beta=1.0,
        **ignored
    ):
    C0 = numpyro.sample("C0", dist.Uniform(min_C0, max_C0))
    x = numpyro.sample("x", dist.Uniform(min_x, max_x))
    beta = numpyro.sample("beta", dist.Uniform(min_beta, max_beta))
    return x, C0, beta

#---

def _sample_sea_xcb_prior_lognormal(
        mean_logx=0.0,
        stdv_logx=1.0,
        mean_logC0=0.0,
        stdv_logC0=1.0,
        mean_logbeta=0.0,
        stdv_logbeta=1.0,
        **ignored
    ):
    x = numpyro.sample("x", dist.LogNormal(mean_logx, stdv_logx))
    C0 = numpyro.sample("C0", dist.LogNormal(mean_logC0, stdv_logC0))
    beta = numpyro.sample("beta", dist.LogNormal(mean_logbeta, stdv_logbeta))
    return x, C0, beta

#---

def _sample_sea_xcb_prior_fancy(
        **ignored
    ):
    # sample in "more independent" basis
    C0 = numpyro.sample('C0', dist.Uniform(0.0, 3.0))   # the co-dimension
    xi3 = numpyro.sample('xi3', dist.Uniform(0.5, 1.5)) # the value of xi(p=3)
    r6 = numpyro.sample('r6', dist.Uniform(0.0, 2.0))   # the ratio of xi(p=6)/xi(p=3)

    beta = numpyro.deterministic('beta', 1 - ((2-r6)*xi3/C0)**0.5)
    x = numpyro.deterministic('x', 1-xi3+C0*(1-beta))
        
    # return
    return x, C0, beta

#---

_sample_sea_xcb_prior = _sample_sea_xcb_prior_uniform
#_sample_sea_xcb_prior = _sample_sea_xcb_prior_lognormal
#_sample_sea_xcb_prior = _sample_sea_xcb_prior_fancy

#------------------------

init_xcb_values = dict( # guesses to land on the mode we want
#    C0=1.00,
#    beta=0.33,
#    x=0.75,
) 

#-----------

def simple_sample_scaling_exponent_ansatz(*args, **kwargs):
    """sample the distribution of scaling exponents using a KDE model of marginal likelihoods
    """
    raise NotImplementedError("""WRITE ME!""")










#-----------

def sample_scaling_exponent_ansatz(
        scales,
        mom,
        cov,
        indexes,
        ref_scale,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        num_retained=DEFAULT_NUM_RETAINED,
        seed=[DEFAULT_SEED],
        verbose=False,
        num_segs=1,
        **prior_kwargs
    ):
    """sample for the distribution of scaling exponents
    """
    if numpyro is None:
        raise ImportError('could not import numpyro')

    if verbose:
        print('defining model')

    num_scales = len(scales)
    num_indexes = len(indexes)

    stdv = jnp.array([np.diag(cov[snd])**0.5 for snd in range(num_scales)], dtype=float)

    def sample_posterior(obs):
        # draw from prior
        x, C0, beta, dlSdls, amp, xi, sl, bl, nl, sh, bh, nh = _sample_sea_prior(indexes, ref_scale, **prior_kwargs)

        # compute expected value of structure function at all indexes for this scale
        sf = _vmap_structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh)

        # compare to observed data (all indexes at this scale)
        ### assume independent uncertainties for each index
        numpyro.sample('mom', dist.Normal(sf, stdv), obs=obs)

        # FIXME?
        # there are extremely strong correlations between mom at the same scale with different indexes
        # this makes it difficult to sample from the joint MultivariateNormal distribution
        # therefore, we fudge this and instead sample from the marginals as if they were independent

#        ### assume correlated uncertainties
#        numpyro.sample('mom', dist.MultivariateNormal(sf, cov[snd]), obs=obs[snd])

    #---

    Prior = None
    Posterior = None

    for s in seed:

        # run the sampler

        if verbose:
            print('running sampler for prior with seed=%d for %d warmup and %d samples' % \
                (s, num_warmup, num_samples))

        mcmc = MCMC(
            NUTS(_sample_sea_prior, init_strategy=init_to_value(values=init_xcb_values)),
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        mcmc.run(random.PRNGKey(s), indexes, ref_scale, **prior_kwargs)

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        prior = mcmc.get_samples()
        prior = thin(num_samples, prior, prior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            prior = dict((key, val[-num_retained:]) for key, val in prior.items())

        if Prior is None:
            Prior = dict((k, [v]) for k, v in prior.items())

        else:
            for k, v in prior.items():
                Prior[k].append(v)

        #---

        if verbose:
            print('running sampler for posterior with seed=%d for %d warmup and %d samples' % \
                (s, num_warmup, num_samples))

        mcmc = MCMC(
            NUTS(sample_posterior, init_strategy=init_to_value(values=init_xcb_values)),
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        mcmc.run(random.PRNGKey(s), mom)

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        posterior = mcmc.get_samples()
        posterior = thin(num_samples, posterior, posterior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            posterior = dict((key, val[-num_retained:]) for key, val in posterior.items())

        # record the likelihood of each sample
        if verbose:
            print('computing likelihood at samples')

        posterior.update(numpyro.infer.log_likelihood(sample_posterior, posterior, mom))

        if Posterior is None:
            Posterior = dict((k, [v]) for k, v in posterior.items())

        else:
            for k, v in posterior.items():
                Posterior[k].append(v)

    #---

    Posterior = dict((k, np.concatenate(tuple(v))) for k, v in Posterior.items())
    Prior = dict((k, np.concatenate(tuple(v))) for k, v in Prior.items())

    if verbose:
        print('\n>>> retained %d total prior samples' % len(list(Prior.values())[0]))
        print('>>> retained %d total posterior samples\n' % len(list(Posterior.values())[0]))

    # return
    return Posterior, Prior

#-------------------------------------------------

def _sample_sfa_prior(
        mean_logamp=-10.0,
        stdv_logamp=10.0,
        mean_xi=0.0,
        stdv_xi=3.0,
        mean_logsl=np.log(10),
        stdv_logsl=1.0,
        mean_bl=0.0,
        stdv_bl=3.0,
        mean_nl=0.0,
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
        seed=[DEFAULT_SEED],
        verbose=False,
        num_segs=1,
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

    Prior = None
    Posterior = None

    for s in seed:

        if verbose:
            print('running sampler for prior with seed=%d for %d warmup and %d samples' % (s, num_warmup, num_samples))

        mcmc = MCMC(NUTS(_sample_sfa_prior), num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(s), **prior_kwargs)

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        prior = mcmc.get_samples()
        prior = thin(num_samples, prior, prior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            prior = dict((key, val[-num_retained:]) for key, val in prior.items())

        if Prior is None:
            Prior = dict((k, [v]) for k, v in prior.items())

        else:
            for k, v in prior.items():
                Prior[k].append(v)

        #---

        if verbose:
            print('running sampler for posterior with seed=%d for %d warmup and %d samples' % \
                (s, num_warmup, num_samples))

        mcmc = MCMC(NUTS(sample_posterior), num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(s), mom)

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        posterior = mcmc.get_samples()
        posterior = thin(num_samples, posterior, posterior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            posterior = dict((key, val[-num_retained:]) for key, val in posterior.items())

        # record the likelihood of each sample

        if verbose:
            print('computing likelihood at samples')

        posterior.update(numpyro.infer.log_likelihood(sample_posterior, posterior, mom))

        if Posterior is None:
            Posterior = dict((k, [v]) for k, v in posterior.items())

        else:
            for k, v in posterior.items():
                Posterior[k].append(v)

    #---

    Posterior = dict((k, np.concatenate(tuple(v))) for k, v in Posterior.items())
    Prior = dict((k, np.concatenate(tuple(v))) for k, v in Prior.items())

    if verbose:
        print('\n>>> retained %d total prior samples' % len(list(Prior.values())[0]))
        print('>>> retained %d total posterior samples\n' % len(list(Posterior.values())[0]))

    # return
    return Posterior, Prior
