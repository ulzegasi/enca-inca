#Author: Firat Ozdemir, October 2019, firat.ozdemir@datascience.ch
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
import numpy as np
from collections import namedtuple


def babcock_leighton_fn(p_n, b_1=0.6, w_1=0.2, b_2=1.0, w_2=0.8):
    '''
    Babcock-Leighton function measuring the efficiency of poloidal field production from the decay of active regions as
    a function of the deep-seated toroidal magnetic component.
    ----
    Function implements f(p_{n}), Eqn 2 in
    FLUCTUATIONS IN BABCOCK-LEIGHTON DYNAMOS. II. REVISITING THE GNEVYSHEV-OHL RULE,
    https://iopscience.iop.org/article/10.1086/511177/pdf,
    Parameter choice b_1=0.6, w_1=0.2, b_2=1.0, w_2=0.8 is based on the above manuscript.'''
    f_p_n = 0.5 * (1. + math.erf((p_n - b_1) / w_1)) * (1. - math.erf((p_n - b_2) / w_2))
    return f_p_n


def babcock_leighton(p_old, alpha, epsilon):
    '''
    Amplitude p_{n+1} of the upcoming cycle.
    -----
    Function implements Eqn 3 in
    FLUCTUATIONS IN BABCOCK-LEIGHTON DYNAMOS. II. REVISITING THE GNEVYSHEV-OHL RULE,
    https://iopscience.iop.org/article/10.1086/511177/pdf'''
    if epsilon < 0.0:
        raise AssertionError('Entered epsilon: %.3f is < 0!' % (epsilon))
    elif epsilon > 1.0:
        logging.warning('Entered epsilon: %.3f is not << 1!' % (epsilon))
    #Note: According to manuscript, epsilon <= 0.39 is the safe zone.
    p_new = alpha * babcock_leighton_fn(p_old) * p_old + epsilon
    return p_new


def uniform_sampler(val_min, val_max, prng=None, num_samples=1):
    '''Function uniformly samples a value btw [val_min, val_max]
        It is highly recommended to provide a pseudorandom number generator for reproducible results.'''
    if prng is None:
        logging.warning('prng not shared with uniform_sampler! Experiments will NOT be reproducible!')
        prng = np.random.RandomState(seed=None)
    return prng.uniform(low=val_min, high=val_max, size=num_samples)

def alpha_sampler(alpha_min, alpha_max, prng=None, num_samples=1):
    '''Function returns sample alpha value(s).'''
    return uniform_sampler(val_min=alpha_min, val_max=alpha_max, prng=prng, num_samples=num_samples)

def epsilon_sampler(epsilon_max, prng=None, num_samples=1):
    '''Function returns sample epsilon value(s).'''
    return uniform_sampler(val_min=0., val_max=epsilon_max, prng=prng, num_samples=num_samples)

def sample_pn_timeseries_v2(p0, alpha_min, alpha_max, epsilon_max, prng=None, len_timeseries=1000):
    '''Function samples a time series of p_n values, then returns computed p_n, sampled alphas and epsilons'''
    #TODO: move to a class obj
    # Pn_Tuple = namedtuple('PN_tuple', ['p', 'a', 'e', 'f'])
    if prng is None:
        logging.warning('sample_pn_timeseries should receive a prng for reproducible results.')
        prng = np.random.RandomState()
    pn = p0
    l_p, l_a, l_e = [], [], []
    l_f = []
    for i in range(len_timeseries):
        a = alpha_sampler(alpha_min=alpha_min, alpha_max=alpha_max, prng=prng)
        e = epsilon_sampler(epsilon_max=epsilon_max, prng=prng)
        f = babcock_leighton_fn(p_n=pn)
        l_p.append(pn)
        pn = babcock_leighton(p_old=pn, alpha=a, epsilon=e)
        l_a.append(a)
        l_e.append(e)
        l_f.append(f)
    p = np.reshape(l_p, (len_timeseries,))
    a = np.reshape(l_a, (len_timeseries,))
    e = np.reshape(l_e, (len_timeseries,))
    f = np.reshape(l_f, (len_timeseries,))
    d = {'p':p, 'a':a, 'e':e, 'f':f}
    # pn_tuple = Pn_Tuple(p=l_p, a=l_a, e=l_e, f=l_f)
    return d










