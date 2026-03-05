# Author: Firat Ozdemir, October 2019, firat.ozdemir@datascience.ch
# Updated by: Simone Ulzega, March 2026, simone.ulzega@zhaw.ch

import numpy as np
import src.utils as utils


class DataGenerator_SolarDynamo_Simplified:
    def __init__(self, len_timeseries=1000, **kwargs):
        self.prng = kwargs.get('prng', np.random.RandomState(seed=1923))
        self.len_timeseries = int(len_timeseries)
        self.p0 = kwargs.get('p0', 1.0)
        # self.p0_std = kwargs.get('p0_std', 0.1)
        self.alpha1_min = kwargs.get('alpha1_min', 0.9)
        self.alpha1_max = kwargs.get('alpha1_max', 1.4)
        self.alpha1_lims = kwargs.get('alpha1_lims', None) # overwrites alpha1_min and alpha1_max
        if self.alpha1_lims is not None:
            self.alpha1_min, self.alpha1_max = self.alpha1_lims[0], self.alpha1_lims[1]
        self.delta_min = kwargs.get('delta_min', 0.05)
        self.delta_max = kwargs.get('delta_max', 0.25)
        self.delta_lims = kwargs.get('delta_lims', None) # overwrites delta_min and delta_max
        if self.delta_lims is not None:
            self.delta_min, self.delta_max = self.delta_lims[0], self.delta_lims[1]
        self.epsilon_min = kwargs.get('epsilon_max', 0.02)
        self.epsilon_max = kwargs.get('epsilon_max', 0.15)
        self.epsilon_lims = kwargs.get('epsilon_lims', None) # overwrites epsilon_min and epsilon_max
        if self.epsilon_lims is not None:
            self.epsilon_min, self.epsilon_max = self.epsilon_lims[0], self.epsilon_lims[1]
        self.alpha1 = kwargs.get('alpha1', None)
        self.delta = kwargs.get('delta', None)
        if self.alpha1 is not None and self.delta is not None:
            self.overwrite_random_alpha = True
            print('Overwriting alpha1 and delta. They will NOT be sampled from a uniform distribution.')
        else:
            self.overwrite_random_alpha = False

    def __iter__(self):
        while True:
            # sample p0, alpha1,alpha2,epsilon
            p0 = self.p0
            if self.overwrite_random_alpha:
                alpha1 = self.alpha1
                delta = self.delta
            else:
                alpha1 = self.prng.uniform(low=self.alpha1_min, high=self.alpha1_max)
                delta = self.prng.uniform(low=self.delta_min, high=self.delta_max)
            epsilon_max = self.prng.uniform(low=self.epsilon_min, high=self.epsilon_max)
            batch = utils.sample_pn_timeseries_v2(p0=p0, alpha_min=alpha1, alpha_max=alpha1+delta, epsilon_max=epsilon_max,
                                                   prng=self.prng, len_timeseries=self.len_timeseries)
            noise_n = np.stack([batch['a'], batch['e']], axis=-1)
            x = np.expand_dims(batch['p'], 1) # [len_timeseries, 1]
            params = (alpha1, delta, epsilon_max) # tuple of scalars
            noise = np.stack([batch['a'], batch['e']], axis=-1) # [timeseries, N]
            d = (x, params, noise)
            yield d

class DataGenerator_SolarDynamo_Simplified_BatchSampler(DataGenerator_SolarDynamo_Simplified):
    def __init__(self, len_timeseries=1000, batch_size=None, **kwargs):
        self.batch_size = batch_size
        super().__init__(len_timeseries=len_timeseries, **kwargs)

    def __iter__(self):
        assert self.batch_size is not None # at this point, batch size must have been already defined.
        while True:
            # sample p0, alpha1,alpha2,epsilon
            p0 = self.p0
            if self.overwrite_random_alpha:
                alpha1 = self.alpha1
                delta = self.delta
            else:
                alpha1 = self.prng.uniform(low=self.alpha1_min, high=self.alpha1_max)
                delta = self.prng.uniform(low=self.delta_min, high=self.delta_max)
            epsilon_max = self.prng.uniform(low=self.epsilon_min, high=self.epsilon_max)
            l_observation = np.empty((self.batch_size, self.len_timeseries, 1)) # shape [batch_size, timeseries, 1]
            l_noise = np.empty((self.batch_size, self.len_timeseries, 2)) # [batch_size, len_timeseries, 2]
            params = (alpha1, delta, epsilon_max) # tuple of scalars
            for i in range(self.batch_size):
                sample = utils.sample_pn_timeseries_v2(p0=p0, alpha_min=alpha1, alpha_max=alpha1+delta, epsilon_max=epsilon_max, \
                    prng=self.prng, len_timeseries=self.len_timeseries)
                l_observation[i,...] = np.expand_dims(sample['p'], 1)
                l_noise[i,...] = np.stack([sample['a'], sample['e']], axis=-1)
            d = (l_observation, params, l_noise)
            yield d

class DataGenerator_NLAR1_Simplified:
    def __init__(self, len_timeseries=200, **kwargs):
        self.prng = kwargs.get('prng', np.random.RandomState(seed=1923))
        self.len_timeseries = int(len_timeseries)
        self.x_0 = kwargs.get('x_0', 0.25)
        self.c_lims = kwargs.get('c_lims', [4.2, 5.8])
        self.sigma_lims = kwargs.get('sigma_lims', [0.005, 0.025])
        self.cast_out_dtype = kwargs.get('cast_out_dtype', None)
        self.func = kwargs.get('fn', None)
        if self.func is not None:
            self.fn = self.func
        else:
            self.fn = lambda c, x_old, sigma, epsilon: c * x_old**2 * (1-x_old) + sigma * epsilon
        
    def draw_c(self):
        if not (isinstance(self.c_lims, tuple) or isinstance(self.c_lims, list)):
            raise AssertionError('c_lims has to be a list or tuple. Found %s.' % (type(self.c_lims)))
        if len(self.c_lims) != 2:
            raise AssertionError('c_lims has to be a list of 2 components [min_c, max_c]. Found c_lims length: %d' % len(self.c_lims))
        return float(self.prng.uniform(low=self.c_lims[0], high=self.c_lims[1]))
            
    def draw_sigma(self):
        if not (isinstance(self.sigma_lims, tuple) or isinstance(self.sigma_lims, list)):
            raise AssertionError('sigma_lims has to be a list or tuple. Found %s.' % (type(self.sigma_lims)))
        if len(self.sigma_lims) != 2: #lower & upper bound
            raise AssertionError('sigma_lims has to be a list of 2 components [min_sigma, max_sigma]. Found sigma_lims length: %d' % len(self.sigma_lims))
        return float(self.prng.uniform(low=self.sigma_lims[0], high=self.sigma_lims[1]))

    def sample_series(self, x0, c0, sigma0):
        d = {'x_i': np.zeros((self.len_timeseries,)),
             'c_i':np.zeros((self.len_timeseries,)),
             'sigma_i': np.zeros((self.len_timeseries,)),
             'epsilon_i': np.zeros((self.len_timeseries,)),
             'c_0': float(c0),
             'sigma_0': float(sigma0),
             'x_0': float(x0),
             }
        x_old = x0
        c = c0
        sigma = sigma0
        for i in range(self.len_timeseries):
            epsilon = self.prng.normal(loc=0, scale=1)
            # if self.cast_out_dtype is not None:
            #     epsilon = np.array(epsilon, dtype=self.cast_out_dtype)
            d['x_i'][i] = x_old
            d['c_i'][i] = c
            d['sigma_i'][i] = sigma
            d['epsilon_i'][i] = epsilon
            # get the next timeseries item.
            x_new = self.fn(c=c, x_old=x_old, sigma=sigma, epsilon=epsilon)
            # update parameters (when needed)
            x_old = x_new
        return d

    def __iter__(self):
        while True:
            x0 = self.x_0
            c0 = self.draw_c()
            sigma0 = self.draw_sigma()
            sample = self.sample_series(x0=x0, c0=c0, sigma0=sigma0)
            # 
            noise = np.expand_dims(sample['epsilon_i'], -1) # shape [timeseries, 1]
            params = (c0, sigma0) # parameters (c, sigma)
            x = np.expand_dims(sample['x_i'], 1) # [len_timeseries, 1]
            if self.cast_out_dtype is not None:
                noise = np.array(noise, dtype=self.cast_out_dtype)
                x = np.array(x, dtype=self.cast_out_dtype)
                params = (np.array(c0, dtype=self.cast_out_dtype), np.array(sigma0, dtype=self.cast_out_dtype))
            d = (x, params, noise)
            yield d

class DataGenerator_NLAR1_Simplified_BatchSampler(DataGenerator_NLAR1_Simplified):
    def __init__(self, len_timeseries=200, batch_size=None, **kwargs):
        self.batch_size = batch_size
        super().__init__(len_timeseries=len_timeseries, **kwargs)

    def __iter__(self):
        assert self.batch_size is not None # at this point, batch size must have been already defined.
        while True:
            x0 = self.x_0
            c0 = self.draw_c()
            sigma0 = self.draw_sigma()
            l_observation = np.empty((self.batch_size, self.len_timeseries, 1)) # shape [batch_size, timeseries, 1]
            l_noise = np.empty((self.batch_size, self.len_timeseries, 1)) # [batch_size, len_timeseries, 1]
            params = (c0, sigma0) # parameters (c, sigma)
            for i in range(self.batch_size):
                sample = self.sample_series(x0=x0, c0=c0, sigma0=sigma0)
                l_observation[i, ...] = np.expand_dims(sample['x_i'], 1)
                l_noise[i,...] = np.expand_dims(sample['epsilon_i'], -1)
            d = (l_observation, params, l_noise)
            yield d


# --- Generators for the Julia SDDE solar dynamo model ---
class DataGenerator_SolarDynamo_SDDE_ENCA:
    """
    Yields tuples (x, params, noise) for ENCA training.

    Deterministic map:
        x = M(theta, eps)
    with eps ~ N(0,1) independent of theta.

    - x:     [L, 1] float32
    - params:[5]    float32  (tau, T, Nd, sigma, Bmax)
    - noise: [L, C] float32  (Gaussian channels fed to decoder)

    where L = Tobs/saveat (requires Tobs divisible by saveat).
    """

    def __init__(
        self,
        Tobs: int = 929,
        saveat: float = 1.0,
        num_noise_channels: int = 1,
        *,
        prng=None,
        Twarmup: int = 200,
        dt: float = 0.1,
        tau_lims=(0.1, 10.0),
        T_lims=(0.1, 10.0),
        Nd_lims=(1.0, 15.0),
        sigma_lims=(0.01, 0.3),
        Bmax_lims=(1.0, 15.0),
    ):
        self.prng = prng if prng is not None else np.random.RandomState(seed=1822)

        self.Tobs = int(Tobs)
        self.saveat = float(saveat)
        self.num_noise_channels = int(num_noise_channels)

        self.Twarmup = int(Twarmup)
        self.dt = float(dt)

        self.tau_lims = tuple(tau_lims)
        self.T_lims = tuple(T_lims)
        self.Nd_lims = tuple(Nd_lims)
        self.sigma_lims = tuple(sigma_lims)
        self.Bmax_lims = tuple(Bmax_lims)

        # enforce clean lengths (avoid silent off-by-one headaches)
        ratio = self.Tobs / self.saveat
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"Tobs ({self.Tobs}) must be divisible by saveat ({self.saveat}).")
        
        # also require Twarmup divisible by saveat
        ratio_w = self.Twarmup / self.saveat
        if abs(ratio_w - round(ratio_w)) > 1e-9:
            raise ValueError(f"Twarmup ({self.Twarmup}) must be divisible by saveat ({self.saveat}).")

        self.L = int(round(ratio))  # output length after downsampling

    def _sample_theta(self):
        tau = float(self.prng.uniform(*self.tau_lims))

        # --- Quantize T to multiples of dt ---
        Tmin, Tmax = self.T_lims
        dt = self.dt

        lag_min = int(np.ceil(Tmin / dt))
        lag_max = int(np.floor(Tmax / dt))
        if lag_max < lag_min:
            raise ValueError(f"T_lims={self.T_lims} incompatible with dt={dt}")

        lag_steps = int(self.prng.randint(lag_min, lag_max + 1))
        T = float(lag_steps * dt)

        Nd = float(self.prng.uniform(*self.Nd_lims))
        sigma = float(self.prng.uniform(*self.sigma_lims))
        Bmax = float(self.prng.uniform(*self.Bmax_lims))
        return (tau, T, Nd, sigma, Bmax)

    def __iter__(self):
        # lazy import (safe wrt Julia/TF init ordering)
        from src.sdde_solar_dynamo_julia import sn_from_noise

        # EM increments: one eps per dt-step over [0, Tsim]
        Tsim = self.Twarmup + self.Tobs
        N_increments = int(round(Tsim / self.dt))
        if abs(N_increments * self.dt - Tsim) > 1e-9:
            raise ValueError(
                f"Tsim ({Tsim}) must be divisible by dt ({self.dt}). "
                f"Got N_increments*dt = {N_increments*self.dt}."
            )

        while True:
            theta = self._sample_theta()

            # bare noise for the Wiener increments (theta-independent)
            eps_dt = self.prng.normal(0.0, 1.0, size=(N_increments,)).astype(np.float32)

            # generate observation (already downsampled by saveat inside Julia path)
            y = sn_from_noise(
                theta,
                eps_dt,
                Twarmup=self.Twarmup,
                Tobs=self.Tobs,
                dt=self.dt,
                saveat=self.saveat,
            )
            y = np.asarray(y, dtype=np.float32)          # [L]
            x = y.reshape(-1, 1)                         # [L, 1]

            params = np.asarray(theta, dtype=np.float32) # [5]

            # decoder noise: match x length L
            if self.num_noise_channels == 1:
                noise = self.prng.normal(0.0, 1.0, size=(self.L, 1)).astype(np.float32)
            else:
                noise = self.prng.normal(0.0, 1.0, size=(self.L, self.num_noise_channels)).astype(np.float32)

            yield (x, params, noise)
            

class DataGenerator_SolarDynamo_SDDE_INCA:
    """
    Yields (Xrep, params) for INCA:

      Xrep:  (Nrep, L, 1) float32   replicas for same theta
      params:(5,) float32           (tau, T, Nd, sigma, Bmax)

    No explicit noise output.
    """
    def __init__(
        self,
        Tobs: int = 929,
        saveat: float = 1.0,
        *,
        prng=None,
        Twarmup: int = 200,
        dt: float = 0.1,
        nrep: int = 8,
        tau_lims=(0.1, 10.0),
        T_lims=(0.1, 10.0),
        Nd_lims=(1.0, 15.0),
        sigma_lims=(0.01, 0.3),
        Bmax_lims=(1.0, 15.0),
    ):
        self.prng = prng if prng is not None else np.random.RandomState(1822)

        self.Tobs = int(Tobs)
        self.saveat = float(saveat)
        self.Twarmup = int(Twarmup)
        self.dt = float(dt)
        self.nrep = int(nrep)

        self.tau_lims = tuple(tau_lims)
        self.T_lims = tuple(T_lims)
        self.Nd_lims = tuple(Nd_lims)
        self.sigma_lims = tuple(sigma_lims)
        self.Bmax_lims = tuple(Bmax_lims)
        
        if abs(self.saveat - 1.0) > 1e-12:
            raise ValueError("This INCA generator currently assumes saveat == 1.0 (matches Julia sn()).")

        ratio = self.Tobs / self.saveat
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"Tobs ({self.Tobs}) must be divisible by saveat ({self.saveat}).")
        self.L = int(round(ratio))

    def _sample_theta(self):
        tau = float(self.prng.uniform(*self.tau_lims))

        # quantize T to dt-grid (same rationale as ENCA generator)
        Tmin, Tmax = self.T_lims
        dt = self.dt
        lag_min = int(np.ceil(Tmin / dt))
        lag_max = int(np.floor(Tmax / dt))
        if lag_max < lag_min:
            raise ValueError(f"T_lims={self.T_lims} incompatible with dt={dt}")
        lag_steps = int(self.prng.randint(lag_min, lag_max + 1))
        T = float(lag_steps * dt)

        Nd = float(self.prng.uniform(*self.Nd_lims))
        sigma = float(self.prng.uniform(*self.sigma_lims))
        Bmax = float(self.prng.uniform(*self.Bmax_lims))
        return (tau, T, Nd, sigma, Bmax)

    def __iter__(self):
        from src.sdde_solar_dynamo_julia import sn_nrep  # lazy import for Julia/TF safety

        while True:
            theta = self._sample_theta()
            params = np.asarray(theta, dtype=np.float32)  # (5,)

            seeds = self.prng.randint(1, 2**31 - 1, size=(self.nrep,)).tolist()  # plain Python ints

            Xrep = sn_nrep(
                theta,
                seeds,
                Twarmup=self.Twarmup,
                Tobs=self.Tobs,
                dt=self.dt,
                saveat=self.saveat,
            )

            Xrep = np.asarray(Xrep, dtype=np.float32)[..., None]   # (Nrep, L, 1)

            yield (Xrep, params)