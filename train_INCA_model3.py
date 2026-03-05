# Author: Simone Ulzega, March 2026, simone.ulzega@zhaw.ch
################
# INCA training pipeline (Implicit Noise Conditional Autoencoder)
# - Encoder maps a *replica set* of time series for same theta to latent summary stats
# - Aggregator maps replica latent summaries to theta_hat
#
# IMPORTANT: init_julia() must happen before importing tensorflow,
# otherwise there will be a conflict in the shared libraries used by both.
from julia_bootstrap import init_julia
init_julia()

import tensorflow as tf
assert tf.__version__.startswith("2."), f"TensorFlow 2.x required, got {tf.__version__}"

import os, glob
import shutil
import datetime
import numpy as np
import sys
import json
import logging
import time

# --- Configuring Python’s logging system ---
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# --- Debugging switch for eager execution ---
_DEBUG = False
if _DEBUG:
    tf.config.run_functions_eagerly(True)

# custom libs
import src.generators
import src.utils_tf


##################################################################################################
class Architecture:
    """
    INCA architecture (Firat style):
    - encoder takes x shaped [B, Nrep, L, 1] and returns z shaped [B, Nrep, D]
    - aggregator takes z shaped [B, Nrep, D] and returns theta_hat shaped [B, P]
    """
    def __init__(self, mb_of_same_realizations, ndims_latent, ndims_free_latent_parameters, len_timeseries, ndims_output):
        self.ndims_latent = int(ndims_latent)
        self.ndims_free_latent_parameters = int(ndims_free_latent_parameters)
        self.len_timeseries = int(len_timeseries)
        self.mb_of_same_realizations = int(mb_of_same_realizations)
        self.num_input_channels = 1
        self.ndims_output = int(ndims_output)

        self.encoder = self.encoder_fn()
        self.aggregator = self.aggregator_fn(num_free_parameters=self.ndims_free_latent_parameters)

    def encoder_fn(self):
        """
        x_input: [B, Nrep, L, 1]
        returns: [B, Nrep, D]
        Uses Firat's ND-conv trick: reshape to 3D for Conv1D / MaxPool1D, then reshape back.
        """
        def maxpool1D_nd_fn(x):
            shp = x.shape
            x = tf.reshape(x, shape=[-1, shp[-2], shp[-1]])          # [B*Nrep, L, C]
            x = tf.keras.layers.MaxPool1D(pool_size=2)(x)           # [B*Nrep, L/2, C]
            x = tf.reshape(x, shape=[-1] + [s for s in shp[1:-2]] + [x.shape[-2], shp[-1]])
            return x

        class Conv1D_ND(tf.keras.layers.Layer):
            def __init__(self, filters, act='relu', **kwargs):
                super().__init__(**kwargs)
                self.filters = int(filters)
                self.act = act

            def build(self, input_shape):
                self.conv = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=3, activation=self.act)

            def call(self, inputs):
                shp = inputs.shape
                x = tf.reshape(inputs, shape=[-1, shp[-2], shp[-1]])  # [B*Nrep, L, C]
                x = self.conv(x)
                x = tf.reshape(x, shape=[-1] + [s for s in shp[1:-2]] + [x.shape[-2], self.filters])
                return x

        x_input = tf.keras.layers.Input(
            shape=[self.mb_of_same_realizations, self.len_timeseries, self.num_input_channels],
            name='x_observation'
        )
        x = x_input

        num_conv_filters = [[16, 16], [32, 32]]
        for i in range(len(num_conv_filters)):
            if i != 0:
                x = tf.keras.layers.Lambda(maxpool1D_nd_fn, name=f"maxpool{i+1}")(x)
            for j, nf in enumerate(num_conv_filters[i]):
                x = Conv1D_ND(filters=nf, act='relu', name=f"conv{i+1}_{j+1}")(x)

        x = Conv1D_ND(filters=self.ndims_latent, act=None, name='final_conv')(x)

        # global avg pool over time dimension (axis=-2), leaving [B, Nrep, D]
        latent_space = tf.keras.layers.Lambda(lambda a: tf.reduce_mean(a, axis=-2), name='global_avg_pool')(x)

        return tf.keras.Model(inputs=x_input, outputs=latent_space, name="encoder")

    def aggregator_fn(self, num_free_parameters):
        """
        latent_mappings: [B, Nrep, D]
        output theta_hat: [B, P]
        Implements Firat's "free parameters" + weighting vector scheme.
        """
        if self.ndims_latent <= num_free_parameters:
            raise AssertionError("ndims_latent <= num_free_parameters: doesn't make sense.")

        latent_mappings = tf.keras.layers.Input(
            shape=[self.mb_of_same_realizations, self.ndims_latent],
            name='latent_representations'
        )

        # choose hidden widths (same heuristic as Firat)
        if num_free_parameters < 2:
            num_ihn = [3, 10, 3]
        elif num_free_parameters < 10:
            num_ihn = [10, 100, 10]
        else:
            raise AssertionError(f"num_free_parameters={num_free_parameters} too large (heuristic).")

        # Slice free dims from the *end*
        x = tf.keras.layers.Lambda(
            lambda t: t[..., self.ndims_latent - num_free_parameters:],
            name='slice_free_parameters'
        )(latent_mappings)

        for i, width in enumerate(num_ihn):
            x = tf.keras.layers.Dense(
                units=width,
                activation=tf.keras.layers.LeakyReLU(negative_slope=0.3),  # avoids alpha deprecation
                name=f'fc_{i+1}'
            )(x)

        weighting_vector = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name=f'fc_{len(num_ihn)+1}'
        )(x)  # [B, Nrep, 1]

        # Clip in a Keras-safe way (no raw tf.reduce_max on KerasTensor)
        def _clip_to_own_max(w):
            wmax = tf.reduce_max(w)
            w = tf.clip_by_value(w, 1e-8, wmax)
            return w

        weighting_vector = tf.keras.layers.Lambda(_clip_to_own_max, name="clip_weights")(weighting_vector)

        # Normalize weights so sum over replicas = 1
        weighting_vector = tf.keras.layers.Lambda(
            lambda w: w / tf.reduce_sum(w, axis=1, keepdims=True),
            name="normalize_weights"
        )(weighting_vector)

        # scale the "model parameter" part (the first D-num_free dims)
        scaled_model_params = tf.keras.layers.Lambda(
            lambda t: t[0][..., :self.ndims_latent - num_free_parameters] * t[1],
            name='scale_model_parameters'
        )([latent_mappings, weighting_vector])  # [B, Nrep, P]

        # weighted average over replicas (Keras-safe)
        theta_hat = tf.keras.layers.Lambda(
            lambda t: tf.reduce_sum(t, axis=1),
            name='pred_model_parameters'
        )(scaled_model_params)  # [B, P]

        return tf.keras.Model(inputs=latent_mappings, outputs=theta_hat, name="aggregator")


##################################################################################################
class Args_:
    def __init__(self, d):
        for k in d.keys():
            setattr(self, k, d[k])


class Manage_Hyper_Parameters:
    """Same as your ENCA version (kept)."""
    def __init__(self, logdir):
        self.param_config_fn = os.path.join(logdir, 'hyper_parameters.json')
        if not os.path.isfile(self.param_config_fn):
            self.args = None
        else:
            with open(self.param_config_fn) as fh:
                args = json.load(fh)
            self.args = Args_(args)
        self.logdir = logdir

    def check_args_maybe_append(self, args):
        if self.args is None:
            return None

        def _norm(v):
            if isinstance(v, (list, tuple)):
                return tuple(v)
            try:
                import numpy as _np
                if isinstance(v, _np.generic):
                    return v.item()
            except Exception:
                pass
            return v

        def _equal(a, b):
            a = _norm(a)
            b = _norm(b)
            if isinstance(a, float) or isinstance(b, float):
                try:
                    return abs(float(a) - float(b)) <= 1e-12
                except Exception:
                    return a == b
            return a == b

        save_args = False
        mismatches = []

        for k in dir(args):
            if k.startswith("__"):
                continue
            if not hasattr(self.args, k):
                print(f'WARNING: key "{k}" missing in saved hyper-parameters; will renew file.')
                save_args = True
                continue
            old_v = getattr(self.args, k)
            new_v = getattr(args, k)
            if not _equal(old_v, new_v):
                mismatches.append((k, old_v, new_v))

        if save_args:
            self.save_parameters(args)

        if mismatches:
            for k, old_v, new_v in mismatches:
                print(f'ERROR! Hyper-parameter mismatch: {k} was {old_v}, now {new_v}.')
            raise RuntimeError(
                f"Hyper-parameter mismatch detected ({len(mismatches)} keys). "
                "Either restore the old settings or start a new logdir."
            )
        return None

    def save_parameters(self, args):
        d_args = {}
        for attr in dir(args):
            if not attr.startswith('__'):
                d_args[attr] = getattr(args, attr)

        d = d_args
        keys = list(d.keys())
        k_drop = []
        for k in keys:
            try:
                json.dumps(d[k])
            except Exception:
                print('dropping key: %s' % k)
                k_drop.append(k)
        for k in k_drop:
            d.pop(k, None)

        with open(self.param_config_fn, 'w') as fh:
            json.dump(d, fh, sort_keys=True, indent=4)

        # copy source file into logdir
        current_script_name = os.path.basename(__file__)
        s_ = str(current_script_name).split('.')
        script_basename = ''.join(s_[:-1])
        script_extension_name = s_[-1]
        src_file = os.path.join(os.getcwd(), current_script_name)
        out_file = os.path.join(self.logdir, current_script_name)
        if os.path.isfile(out_file):
            date = datetime.datetime.now()
            datestr = date.strftime('%Y%m%d')
            out_file = os.path.join(self.logdir, script_basename + datestr + '.' + script_extension_name)
            if os.path.isfile(out_file):
                datestr = date.strftime('%Y%m%d%H%M%S')
                out_file = os.path.join(self.logdir, script_basename + datestr + '.' + script_extension_name)
        shutil.copyfile(src_file, out_file)


##################################################################################################
class ExpSetup:
    def __init__(self):
        tag = "test"
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logdir = os.path.join(
            os.getcwd(),
            "sdde_INCA_runs",
            f"{run_id}_{tag}"
        )

        # INCA core settings
        self.nrep = 8                      # Nrep (replicas per theta)
        self.ndims_latent = 10             # D (latent dims)
        self.num_model_parameters = 5      # P = (tau, T, Nd, sigma, Bmax)
        self.ndims_free_latent_parameters = self.ndims_latent - self.num_model_parameters
        if self.ndims_free_latent_parameters < 1:
            raise ValueError("ndims_latent must be > num_model_parameters for INCA (needs free dims).")

        # loss weights (Firat-style; theta gets heavier weight)
        self.lambda_theta = float(self.nrep)  # common choice
        self.lambda_z = 1.0

        # SDDE sim settings
        self.Twarmup = 200
        self.Tobs = 929
        self.dt = 0.1
        self.saveat = 1.0

        ratio = self.Tobs / self.saveat
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"Tobs ({self.Tobs}) must be divisible by saveat ({self.saveat}).")
        self.len_timeseries = int(round(ratio))

        # training
        self.batch_size = 32                 # B (different thetas per step)
        self.max_training_steps = int(3e6)
        self.freq_log = 100

        # priors
        self.tau_lims = (0.1, 10.0)
        self.T_lims = (0.1, 10.0)
        self.Nd_lims = (1.0, 15.0)
        self.sigma_lims = (0.01, 0.3)
        self.Bmax_lims = (1.0, 15.0)
        
        if self.ndims_latent <= self.num_model_parameters:
            raise ValueError("Need ndims_latent > num_model_parameters (at least 1 free dim).")


##################################################################################################
def main():
    args = ExpSetup()

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    ##################################################################################################
    # Generator (INCA): yields (xrep, params) with xrep: [Nrep, L, 1]
    gen_train = src.generators.DataGenerator_SolarDynamo_SDDE_INCA(
        Tobs=args.Tobs,
        saveat=args.saveat,
        Twarmup=args.Twarmup,
        dt=args.dt,
        nrep=args.nrep,
        tau_lims=args.tau_lims,
        T_lims=args.T_lims,
        Nd_lims=args.Nd_lims,
        sigma_lims=args.sigma_lims,
        Bmax_lims=args.Bmax_lims,
    )
    gen_iter = iter(gen_train)

    def next_batch_from_generator(gen, batch_size):
        """Collect batch_size theta-groups: xrep:(Nrep,L,1), params:(P,)"""
        xs = []
        ps = []

        for _ in range(batch_size):
            xrep, p = next(gen)
            xs.append(xrep)
            ps.append(p)

        x_np = np.stack(xs, axis=0).astype(np.float32)   # [B, Nrep, L, 1]
        p_np = np.stack(ps, axis=0).astype(np.float32)   # [B, P]

        x = tf.convert_to_tensor(x_np)
        params = tf.convert_to_tensor(p_np)

        return x, params


    ##################################################################################################
    # Define the architecture
    model_obj = Architecture(
        mb_of_same_realizations=args.nrep,
        ndims_latent=args.ndims_latent,
        ndims_free_latent_parameters=args.ndims_free_latent_parameters,
        len_timeseries=args.len_timeseries,
        ndims_output=args.num_model_parameters,
    )

    model_obj.encoder.summary()
    model_obj.aggregator.summary()


    ##################################################################################################
    # Loss functions (INCA-consistent shapes)

    class ChiSquareStatistic(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            sd = tf.math.squared_difference(y_true, y_pred)
            denom = tf.maximum(tf.math.pow(y_true, 2), 1e-6)
            return tf.reduce_sum(sd / denom, axis=-1)


    @tf.function
    def loss_theta_fn(params, theta_pred):
        """
        params, theta_pred: [B, P]
        Returns scalar.
        """
        # ChiSquareStatistic returns [B]
        chi = ChiSquareStatistic()(params, theta_pred)
        return tf.reduce_mean(chi)


    @tf.function
    def loss_z_fn(params, z_latent):
        """
        params:   [B, P]
        z_latent: [B, Nrep, D]
        We regress theta into the first P latent dims for each replica.
        Returns scalar.
        """
        P = tf.shape(params)[-1]                     # P
        params_ = tf.expand_dims(params, axis=1)     # [B, 1, P]
        z_ = z_latent[:, :, :P]                      # [B, Nrep, P]

        # ChiSquareStatistic returns [B, Nrep]
        chi = ChiSquareStatistic()(params_, z_)
        return tf.reduce_mean(chi)


    ##################################################################################################
    # Optimizer

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=int(6e3),
        decay_rate=0.92,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    global_gradient_clipnorm = 1e5


    ##################################################################################################
    # Checkpoints

    ckpt = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=model_obj.encoder,
        aggregator=model_obj.aggregator
    )

    save_manager = tf.train.CheckpointManager(
        checkpoint=ckpt,
        directory=args.logdir,
        max_to_keep=3,
        checkpoint_name='model_ckpt'
    )

    save_manager_best = tf.train.CheckpointManager(
        checkpoint=ckpt,
        directory=args.logdir,
        max_to_keep=3,
        checkpoint_name='model_best_ckpt'
    )

    save = lambda sm, ckpt_number=None: sm.save(checkpoint_number=ckpt_number)

    hp_manager = Manage_Hyper_Parameters(logdir=args.logdir)


    ##################################################################################################
    # Restore checkpoint if exists

    ckpt.restore(save_manager.latest_checkpoint)

    if save_manager.latest_checkpoint:
        print(f"Restored from {save_manager.latest_checkpoint}")
        hp_manager.check_args_maybe_append(args)
    else:
        print(f"Initializing training from scratch.\nLogdir: {args.logdir}")
        hp_manager.save_parameters(args)


    ##################################################################################################
    # TensorBoard

    summary_writer = tf.summary.create_file_writer(
        logdir=os.path.join(args.logdir, 'train')
    )

    def export_summary_scalars(d, step, writer):
        with writer.as_default():
            for k, v in d.items():
                tf.summary.scalar(k, v, step=step)

    def export_summary_histograms(d, step, writer):
        with writer.as_default():
            for k, v in d.items():
                tf.summary.histogram(k, v, step=step)


    ##################################################################################################
    # Train step

    @tf.function
    def train_step(model, x, params, optimizer):

        with tf.GradientTape() as tape:

            z_latent = model.encoder(x, training=True)
            theta_pred = model.aggregator(z_latent, training=True)

            loss_theta = loss_theta_fn(params, theta_pred)  # scalar
            loss_z     = loss_z_fn(params, z_latent)        # scalar

            # keep dicts empty for now (or remove them entirely)
            dict_theta = {}
            dict_z = {}

            loss = args.lambda_theta * loss_theta + args.lambda_z * loss_z

        vars_ = (
            model.encoder.trainable_variables
            + model.aggregator.trainable_variables
        )

        grads = tape.gradient(loss, vars_)

        if global_gradient_clipnorm is not None:
            grads, _ = tf.clip_by_global_norm(grads, global_gradient_clipnorm)

        optimizer.apply_gradients(zip(grads, vars_))

        return (loss_theta, loss_z), (theta_pred, z_latent), (dict_theta, dict_z)


    ##################################################################################################
    # Metrics

    avg_loss_total = tf.keras.metrics.Mean()
    avg_loss_theta = tf.keras.metrics.Mean()
    avg_loss_z = tf.keras.metrics.Mean()

    avg_loss_long_term = tf.keras.metrics.Mean()

    curr_best_loss = np.inf


    ##################################################################################################
    # Warm-up

    print("Building first batch...")
    x, params = next_batch_from_generator(gen_iter, args.batch_size)

    print("Warm-up train_step...")
    _ = train_step(model_obj, x, params, optimizer)


    ##################################################################################################
    # Training loop

    step = 0

    while True:

        x, params = next_batch_from_generator(gen_iter, args.batch_size)

        loss_tuple, outputs, dicts = train_step(
            model=model_obj,
            x=x,
            params=params,
            optimizer=optimizer
        )

        theta_pred, z_latent = outputs
        loss_theta, loss_z = loss_tuple

        loss_total = args.lambda_theta * loss_theta + args.lambda_z * loss_z

        tf.debugging.assert_all_finite(loss_total, "loss_total has NaN")

        step = int(optimizer.iterations.numpy())

        if step >= args.max_training_steps:
            logging.info("Training completed.")
            break


        avg_loss_theta.update_state(loss_theta)
        avg_loss_z.update_state(loss_z)
        avg_loss_total.update_state(loss_total)
        avg_loss_long_term.update_state(loss_total)


        ##################################################################
        # Logging

        if step % args.freq_log == 0:

            d_scalars = {
                "loss_total": avg_loss_total.result(),
                "loss_theta": avg_loss_theta.result(),
                "loss_z": avg_loss_z.result(),
                "lr": lr_schedule(step)
            }

            export_summary_scalars(d_scalars, step, summary_writer)

            logging.info(
                f"Step {step}: total={float(d_scalars['loss_total']):.3f} "
                f"theta={float(d_scalars['loss_theta']):.3f} "
                f"z={float(d_scalars['loss_z']):.3f}"
            )

            avg_loss_total.reset_state()
            avg_loss_theta.reset_state()
            avg_loss_z.reset_state()

            d_histograms = dict(
                [(f"enc_{v.name}", tf.identity(v)) for v in model_obj.encoder.trainable_variables] +
                [(f"agg_{v.name}", tf.identity(v)) for v in model_obj.aggregator.trainable_variables]
            )

            export_summary_histograms(d_histograms, step, summary_writer)

            save(save_manager, ckpt_number=step)


        ##################################################################
        # Best model tracking

        if step % (10 * args.freq_log) == 0:

            long_loss = float(avg_loss_long_term.result().numpy())

            if long_loss < curr_best_loss:
                curr_best_loss = long_loss
                avg_loss_long_term.reset_state()

                logging.info(f"New best long-term loss: {curr_best_loss:.3f}")

                save(save_manager_best, ckpt_number=step)


    ##################################################################################################
    save(save_manager, ckpt_number=step)


##################################################################################################
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs visible to TensorFlow:", gpus)
    main()
    
