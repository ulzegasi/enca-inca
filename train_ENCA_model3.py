#Author: Simone Ulzega, February 2026, simone.ulzega@zhaw.ch
################
# This training pipeline has an encoder-decoder-like architecture. From input timeseries to sufficient statistics space, then
# using the noise vectors back to the reconstruction of the initial input signal.

# IMPORTANT: init_julia() must happen before importing tensorflow, 
# otherwise there will be a conflict in the shared libraries used by both.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from julia_bootstrap import init_julia
init_julia()

import tensorflow as tf
assert tf.__version__.startswith("2."), f"TensorFlow 2.x required, got {tf.__version__}"
tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

import glob
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
    ''' Use this class to customize the architecture to be used. 
    This example uses a fully convolutional encoder and a Bidirectional-LSTM-based decoder.'''
    def __init__(self, ndims_latent, len_timeseries, num_noise_channels):
        self.ndims_latent = ndims_latent
        self.len_timeseries = len_timeseries
        self.num_input_channels = 1 #assuming a given observed timeseries is a single channel (vector)
        self.num_noise_channels = num_noise_channels

        self.encoder = self.encoder_fn()
        self.decoder = self.decoder_fn()

    def encoder_fn(self):
        '''x_input size: [bs, #len_timeseries, #num_input_channels] 
        Implements encoder of only convolutional and maxpooling operators'''
        conv_fn = lambda filters, act=None, name=None: tf.keras.layers.Conv1D(filters=filters, kernel_size=3, activation=act, name=name)
        x_input = tf.keras.layers.Input(shape=[self.len_timeseries, self.num_input_channels], name='x_observation')
        x = x_input
        self.num_conv_filters = [[16, 16], [32, 32]]
        for i in range(len(self.num_conv_filters)):
            if i != 0:
                x = tf.keras.layers.MaxPool1D(pool_size=2, name='maxpool%d'%(i+1))(x)
            for j in range(len(self.num_conv_filters[i])):
                x = conv_fn(filters=self.num_conv_filters[i][j], act='relu', name='conv%d_%d'%((i+1), (j+1)))(x) #[batch_size, len_timeseries, num_conv_filters[-1]]
        x = conv_fn(filters=self.ndims_latent, act=None, name='final_conv')(x)  # [batch_size, len_timeseries, ndims_latent]
        latent_space = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        return tf.keras.Model(inputs=x_input, outputs=latent_space)

    def decoder_fn(self):
        '''latent_mappings size: [bs, #ndims_latent]
        noise_vectors size: [bs, #len_timeseries, #num_noise_channels]
        output: [bs, #len_timeseries, #input_channels]'''
        # tile latent_mappings to timeseries length of the noise vectors.
        latent_mappings = tf.keras.layers.Input(shape=[self.ndims_latent], name='latent_representations')
        noise_vectors = tf.keras.layers.Input(shape=[self.len_timeseries, self.num_noise_channels], name='noise_vectors')
        tile_ldims_layer = tf.keras.layers.Lambda(function=lambda x: tf.tile(tf.expand_dims(x, axis=1), multiples=[1, self.len_timeseries, 1]), name='tile_latent_space') 
        concat_inputs = tf.keras.layers.Concatenate(axis=-1, name='concatenate_noise_and_latent_dims')([tile_ldims_layer(latent_mappings), noise_vectors])
        num_units = 16
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_units, return_sequences=True, dtype=tf.float32, name='lstm_cell_1'), name='Bi-cell-1')(concat_inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_units, return_sequences=True, dtype=tf.float32, name='lstm_cell_2'), name='Bi-cell-2')(x)
        x = tf.keras.layers.Dense(units=self.num_input_channels, activation=None, name='pred')(x)
        x = tf.keras.layers.Reshape([self.len_timeseries, self.num_input_channels], name='output_shape')(x)
        return tf.keras.Model(inputs=(latent_mappings, noise_vectors), outputs=x)

##################################################################################################
class Args_:
    def __init__(self, d):
        for k in d.keys():
            setattr(self, k, d[k])

class Manage_Hyper_Parameters:
    '''A simple class to manage everything regarding hyper parameter setup of an experiment. Warn for modifications in the experiment setup if training was interrupted.'''
    def __init__(self, logdir):
        self.param_config_fn = os.path.join(logdir, 'hyper_parameters.json')
        if not os.path.isfile(self.param_config_fn):
            self.args = None
        else:
            with open(self.param_config_fn) as fh:
                args = json.load(fh)
            self.args = Args_(args) # convert dictionary to a class with attributes (to match code similar with training)
        self.logdir = logdir
    def check_args_maybe_append(self, args):
        """
        Compare current args (ExpSetup) against the saved hyper_parameters.json.

        - Treat list/tuple as equivalent (JSON turns tuples into lists).
        - Treat numpy scalars as Python scalars.
        - For floats, allow tiny numerical differences.
        - If a real mismatch is found, raise (so training truly stops).
        - If new keys are present, renew the JSON file.
        """
        if self.args is None:
            return None

        def _norm(v):
            # JSON loads tuples as lists; treat them equivalently.
            if isinstance(v, (list, tuple)):
                return tuple(v)
            # numpy scalars -> python scalars
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

            # float-ish comparisons with tolerance
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
                print(f'WARNING: key "{k}" was missing in the saved hyper-parameter configuration; will renew file.')
                save_args = True
                continue

            old_v = getattr(self.args, k)
            new_v = getattr(args, k)

            if not _equal(old_v, new_v):
                mismatches.append((k, old_v, new_v))

        if save_args:
            self.save_parameters(args)

        if mismatches:
            # Print all mismatches, then stop.
            for k, old_v, new_v in mismatches:
                print(f'ERROR! Mismatch in hyper-parameter settings file. Parameter {k} was {old_v}, now it is {new_v}.')
            raise RuntimeError(
                f"Hyper-parameter mismatch detected ({len(mismatches)} keys). "
                "Either restore the old settings or start a new logdir."
            )

        return None
    def save_parameters(self, args):
        d_args = {}
        for attr in dir(args):
            if not attr.startswith('__'):  # don't get methods
                d_args[attr] = getattr(args, attr)
        d = d_args
        keys = d.keys()
        k_drop = []
        for k in keys:
            try:
                json.dumps(d[k])
            except:
                print('dropping key: %s' % k)
                k_drop.append(k)
        for k in k_drop:
            d.pop(k, None)
        with open(self.param_config_fn, 'w') as fh:
            json.dump(d, fh, sort_keys=True, indent=4)
        # Copy source file (remove this line if source filename changes.)
        current_script_name = os.path.basename(__file__)
        s_ = str(current_script_name).split('.')
        script_basename = ''.join(s_[:-1]) # merge list of strings (in case multiple . exist in filename.)
        script_extension_name = s_[-1]
        src_file = os.path.join(os.getcwd(), current_script_name)
        out_file = os.path.join(self.logdir, current_script_name)
        if os.path.isfile(out_file): # if outfile already exists, save another copy.
            date = datetime.datetime.now()
            datestr = date.strftime('%Y%m%d')
            out_file = os.path.join(self.logdir, script_basename+datestr+'.'+script_extension_name)
            if os.path.isfile(out_file): #if there is already a copy from that day, add time details also.
                datestr = date.strftime('%Y%m%d%H%M%S')
                out_file = os.path.join(self.logdir, script_basename+datestr+'.'+script_extension_name)
        shutil.copyfile(src_file, out_file)

##################################################################################################
class ExpSetup:
    def __init__(self):
        tag = "smoke"   # change this when you test something new
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logdir = os.path.join(
            os.getcwd(),
            "sdde_ENCA_runs",
            f"{run_id}_{tag}"
        )

        self.ndims_latent = 10
        self.num_noise_channels = 1
        self.num_model_parameters = 5  # (tau, T, Nd, sigma, Bmax)

        # SDDE sim settings
        self.Twarmup = 200
        self.Tobs = 271 # C14 dataset: 929, obsSN dataset: 271
        self.dt = 0.1
        self.saveat = 1.0

        # derived length used by NN + dataset shapes
        ratio = self.Tobs / self.saveat
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"Tobs ({self.Tobs}) must be divisible by saveat ({self.saveat}).")
        self.len_timeseries = int(round(ratio))  # equals Tobs if saveat=1.0

        self.batch_size = 64
        self.max_training_steps = int(2500) # int(3e6)
        self.freq_log = 100
        
        # parameter priors
        self.tau_lims = (0.1, 10.0)
        self.T_lims = (0.1, 10.0)
        self.Nd_lims = (1.0, 15.0)
        self.sigma_lims = (0.01, 0.3)
        self.Bmax_lims = (1.0, 15.0)
        
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
    # Define a generator function for observations, parameters, and noise vectors
    gen_train = src.generators.DataGenerator_SolarDynamo_SDDE_ENCA(
        Tobs=args.Tobs,
        saveat=args.saveat,
        num_noise_channels=args.num_noise_channels,
        Twarmup=args.Twarmup,
        dt=args.dt,
        # parameter priors
        tau_lims=args.tau_lims,
        T_lims=args.T_lims,
        Nd_lims=args.Nd_lims,
        sigma_lims=args.sigma_lims,
        Bmax_lims=args.Bmax_lims,
    )
    
    def next_batch_from_generator(gen, batch_size):
        """Collect batch_size samples from the Python generator (main thread)."""
        xs = []
        ps = []
        ns = []
        for _ in range(batch_size):
            x0, p0, n0 = next(gen)   # gen yields numpy arrays
            xs.append(x0)
            ps.append(p0)
            ns.append(n0)

        # Stack into numpy batches
        x_np = np.stack(xs, axis=0).astype(np.float32)      # (B, len, 1)
        p_np = np.stack(ps, axis=0).astype(np.float32)      # (B, 5)
        n_np = np.stack(ns, axis=0).astype(np.float32)      # (B, len, nc)

        # Convert to TF tensors
        x = tf.convert_to_tensor(x_np)
        params = tf.convert_to_tensor(p_np)
        noise = tf.convert_to_tensor(n_np)
        return x, params, noise

    ##################################################################################################
    # Define the architecture
    model_obj = Architecture(ndims_latent=args.ndims_latent, len_timeseries=args.len_timeseries, num_noise_channels=args.num_noise_channels)
    model_obj.encoder.summary()
    model_obj.decoder.summary()

    ##################################################################################################
    # Define objective functions
    class ChiSquareStatistic(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            '''Implements \sum( (y_true - y_pred) ** 2 / y_true'''
            sd = tf.math.squared_difference(y_true, y_pred)
            return tf.math.reduce_sum(sd / tf.math.maximum(tf.math.pow(y_true, 2), 1e-6), axis= -1)


    @tf.function
    def loss_reconstruction_fn(x, x_pred, return_each_dim=False):
        if return_each_dim:
            d = {}
            for i in range(x.shape[-1]): # do over all channels of observation (it's just 1 in our example.)
                # d['MSE_x_ch_%d'%(i+1)] = tf.keras.losses.MeanSquaredError(name='MSE')(x[...,i],x_pred[...,i]) # default behavior is sum over batch size: (average over minibatch size)
                d['ChiSquare_x_ch_%d'%(i+1)] = ChiSquareStatistic(name='ChiSquare')(y_true=x[...,i], y_pred=x_pred[...,i]) / args.len_timeseries # take average of ChiSquare across the timeseries elements (as to scale similarly with ChiSquare of latent parameters)
            return d
        else:
            # return tf.keras.losses.MeanSquaredError(name='MSE')(x,x_pred)
            return ChiSquareStatistic(name='ChiSquare')(x,x_pred) / args.len_timeseries 
    @tf.function
    def loss_regress_params_fn(params, params_pred, return_each_dim=False):
        ''' Notice params_pred can have more dimensions than params (free dimensions).
        If not using this loss, return 0. instead.'''
        num_params = params._shape_as_list()[-1]
        if return_each_dim:
            d = {}
            for i in range(num_params):
                # d['MSE_z_%d'%(i+1)] = tf.keras.losses.MeanSquaredError(name='loss_param_%d'%(i+1))(params[...,i], params_pred[...,i])
                d['ChiSquare_z_%d'%(i+1)] = ChiSquareStatistic(name='ChiSquare_z_%d'%(i+1))(params[...,i], params_pred[...,i]) / num_params
            return d
        else:
            loss = 0.
            for i in range(num_params):
                # loss += tf.keras.losses.MeanSquaredError(name='loss_param_%d'%(i+1))(params[...,i], params_pred[...,i])
                loss += ChiSquareStatistic(name='ChiSquare_z_%d'%(i+1))(params[...,i], params_pred[...,i]) / num_params
            return loss
    # Define additional metrics for logging at every

    ##################################################################################################
    # Define optimizer 
    # --- Custom LR schedule: linear warmup from ~0 to your initial LR, then exponential decay ---
    # lr_schedule = src.utils_tf.LearningRateScheduleExponentialDecayWithLinearWarmup(steps_warmup=args.linear_warmup_steps, initial_learning_rate=1.e-3, decay_steps=int(6*1e3), decay_rate=0.92, staircase=True)
    # --- Exponential decay with fixed initial LR ---
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.e-3, decay_steps=int(6*1e3), decay_rate=0.92, staircase=True)
    # --- Optimizer --- 
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) 
    #, clipnorm=1e5, clipvalue=1.) 
    # #clips |gradients| above clipvalue to prevent exploding., 
    # #clipnorm: preventative measure for divergence. 
    # Unfortunately both are clipping individually for each gradient, which can change the direction of the gradients..
    global_gradient_clipnorm = 1e5

    ##################################################################################################
    # Define ckpt managers to save model weights throughout optimization
    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=model_obj.encoder, decoder=model_obj.decoder)
    save_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=args.logdir, max_to_keep=3, checkpoint_name='model_ckpt')
    save_manager_best = tf.train.CheckpointManager(checkpoint=ckpt, directory=args.logdir, max_to_keep=3, checkpoint_name='model_best_ckpt') # manager for early stopping.
    save = lambda save_manager, ckpt_number=None: save_manager.save(checkpoint_number=ckpt_number)
    # Define a manager object for training hyper-parameters
    hp_manager = Manage_Hyper_Parameters(logdir=args.logdir)

    ##################################################################################################
    # Restore a previously interrupted training session (if exists)
    ckpt.restore(save_manager.latest_checkpoint)
    if save_manager.latest_checkpoint:
        print(f"Restored from {save_manager.latest_checkpoint}.")
        # Double check if experiment setup matches the one in the saved chkpt dir
        hp_manager.check_args_maybe_append(args)
    else:
        print("Initializing training from scratch. \nLogdir: %s" % args.logdir)
        # Dump hyper-parameters to ckpt dir for future reference. 
        hp_manager.save_parameters(args)
        
    ##################################################################################################
    # Setup summaries for tensorboard
    summary_writer = tf.summary.create_file_writer(logdir=os.path.join(args.logdir, 'train'))
    
    def export_summary_scalars(dict_name_and_val, step, writer):
        if not isinstance(dict_name_and_val, dict):
            raise AssertionError('dict_name_and_val must be a dictionary.')
        with writer.as_default():
            for k, v in dict_name_and_val.items():
                tf.summary.scalar(k, v, step=step)

    def export_summary_histograms(dict_name_and_val, step, writer):
        if not isinstance(dict_name_and_val, dict):
            raise AssertionError('dict_name_and_val must be a dictionary.')
        with writer.as_default():
            for k, v in dict_name_and_val.items():
                tf.summary.histogram(k, v, step=step)

    ##################################################################################################
    # wrap a training step for performance gains.
    @tf.function
    def train_step(model, x, params, noise, optimizer):
        """Single training step (no logging inside)."""
        with tf.GradientTape(persistent=False) as tape:
            z_latent = model.encoder(x, training=True)
            x_reconst = model.decoder((z_latent, noise), training=True)

            dict_reconstruction = loss_reconstruction_fn(x, x_reconst, return_each_dim=True)
            loss_reconstruction = tf.reduce_sum(list(dict_reconstruction.values()))

            dict_regression = loss_regress_params_fn(params, z_latent, return_each_dim=True)
            loss_regress_params = tf.reduce_sum(list(dict_regression.values()))

            loss = loss_reconstruction + loss_regress_params

        trainable_variables = model.encoder.trainable_variables + model.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        if global_gradient_clipnorm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clipnorm)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return (loss_reconstruction, loss_regress_params), (z_latent, x_reconst), (dict_reconstruction, dict_regression)
    ##################################################################################################
    # Define a few metrics to export to tensorboard
    avg_loss_total = tf.keras.metrics.Mean(name='loss_total', dtype=tf.float32) # variable will keep track of average of loss value since last freq_log
    avg_loss_recon = tf.keras.metrics.Mean(name='loss_reconstruction', dtype=tf.float32)
    avg_loss_reg_p = tf.keras.metrics.Mean(name='loss_regress_params', dtype=tf.float32)
    dict_avg_loss_recon_items = {} #will create mean metric on the fly since we don't know names in advance.
    dict_avg_loss_reg_p_items = {}
    dict_avg_rmse_recon_items = {}
    dict_avg_rmse_reg_p_items = {}

    ##################################################################################################
    # Define a metric to keep track of best reconstruction over a longer window
    avg_loss_long_term = tf.keras.metrics.Mean(name='loss_long_term', dtype=tf.float32)
    curr_best_loss = np.inf

    ##################################################################################################
    # Execute training loop
    step = 0  # will survive after loop
    
    gen_iter = iter(gen_train)
    
    print("Building first batch...")
    t0 = time.time()
    x, params, noise = next_batch_from_generator(gen_iter, args.batch_size)
    print(f"First batch built in {time.time()-t0:.1f}s: x={x.shape}, params={params.shape}, noise={noise.shape}")

    # --- quick finiteness + stats checks (on first batch) ---
    def assert_finite(name, t):
        t_np = t.numpy()
        if not np.isfinite(t_np).all():
            bad = np.where(~np.isfinite(t_np))
            raise ValueError(f"{name} has non-finite values at {bad[:3]}")

    assert_finite("x", x)
    assert_finite("params", params)
    assert_finite("noise", noise)
    print("Batch finiteness: OK")

    print("x stats:", float(tf.reduce_min(x)), float(tf.reduce_max(x)), float(tf.reduce_mean(x)))
    print("params stats:", float(tf.reduce_min(params)), float(tf.reduce_max(params)), float(tf.reduce_mean(params)))
    print("noise stats:", float(tf.reduce_min(noise)), float(tf.reduce_max(noise)), float(tf.reduce_mean(noise)))

    # --- warm up tf.function tracing/compilation ---
    print("Warming up train_step (tf.function tracing/compilation)...")
    t1 = time.time()
    _ = train_step(model=model_obj, x=x, params=params, noise=noise, optimizer=optimizer)
    print(f"Warmup train_step done in {time.time()-t1:.2f}s")

    # ---------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------
    while True:
        # 1) build batch (python)
        x, params, noise = next_batch_from_generator(gen_iter, args.batch_size)

        # 2) train step (tf.function)
        loss_tuple, z_and_x, dict_mse = train_step(
            model=model_obj, x=x, params=params, noise=noise, optimizer=optimizer
        )

        # unpack
        z_latent, x_reconst = z_and_x
        loss_reconstruction, loss_regress_params = loss_tuple
        loss_total = loss_reconstruction + loss_regress_params
        dict_rec, dict_reg = dict_mse

        # 3) NaN/Inf guard
        tf.debugging.assert_all_finite(loss_total, "loss_total has NaN or Inf")

        # 4) step / stopping
        step = int(optimizer.iterations.numpy())
        if step >= args.max_training_steps:
            logging.info("Training is completed.")
            break

        # 5) metrics + logging + checkpoints

        # Update metrics (your existing code continues...)
        avg_loss_recon.update_state(loss_reconstruction)
        avg_loss_reg_p.update_state(loss_regress_params)
        avg_loss_total.update_state(loss_total)
        avg_loss_long_term.update_state(loss_total)

        # Per-channel reconstruction losses
        for k in dict_rec:
            if k not in dict_avg_loss_recon_items:
                dict_avg_loss_recon_items[k] = tf.keras.metrics.Mean(name=k, dtype=tf.float32)
            dict_avg_loss_recon_items[k].update_state(dict_rec[k])

        # Per-parameter regression losses
        for k in dict_reg:
            if k not in dict_avg_loss_reg_p_items:
                dict_avg_loss_reg_p_items[k] = tf.keras.metrics.Mean(name=k, dtype=tf.float32)
            dict_avg_loss_reg_p_items[k].update_state(dict_reg[k])

        # RMSE reconstruction
        for i_ in range(x.shape[-1]):
            k = f'RMSE_x_ch_{i_+1}'
            if k not in dict_avg_rmse_recon_items:
                dict_avg_rmse_recon_items[k] = tf.keras.metrics.RootMeanSquaredError(
                    name='rmse_reconstruction', dtype=tf.float32
                )
            dict_avg_rmse_recon_items[k].update_state(
                y_true=x[..., i_], y_pred=x_reconst[..., i_]
            )

        # RMSE regression
        for i_ in range(params.shape[-1]):
            k = f'RMSE_z_ch_{i_+1}'
            if k not in dict_avg_rmse_reg_p_items:
                dict_avg_rmse_reg_p_items[k] = tf.keras.metrics.RootMeanSquaredError(
                    name='rmse_regularization', dtype=tf.float32
                )
            dict_avg_rmse_reg_p_items[k].update_state(
                y_true=params[..., i_], y_pred=z_latent[..., i_]
            )

        # ------------------------------------------------
        # Logging block
        # ------------------------------------------------
        if step % args.freq_log == 0:

            d_scalars = {
                'loss_total': avg_loss_total.result(),
                'loss_reconstruction': avg_loss_recon.result(),
                'loss_regress_params': avg_loss_reg_p.result(),
            }

            # --- Parameter ranges in current batch ---
            # (safe even with batch_size=1; then min==max)
            p = params.numpy()   # shape (B, 5)

            tau_min, tau_max     = float(p[:,0].min()), float(p[:,0].max())
            T_min, T_max         = float(p[:,1].min()), float(p[:,1].max())
            Nd_min, Nd_max       = float(p[:,2].min()), float(p[:,2].max())
            sigma_min, sigma_max = float(p[:,3].min()), float(p[:,3].max())
            Bmax_min, Bmax_max   = float(p[:,4].min()), float(p[:,4].max())

            logging.info(
                "params ranges: "
                f"tau[{tau_min:.3f},{tau_max:.3f}] "
                f"T[{T_min:.3f},{T_max:.3f}] "
                f"Nd[{Nd_min:.3f},{Nd_max:.3f}] "
                f"sigma[{sigma_min:.3f},{sigma_max:.3f}] "
                f"Bmax[{Bmax_min:.3f},{Bmax_max:.3f}]"
            )

            # Optional: also push ranges to TensorBoard
            d_scalars.update({
                "theta/tau_min": tau_min, "theta/tau_max": tau_max,
                "theta/T_min": T_min, "theta/T_max": T_max,
                "theta/Nd_min": Nd_min, "theta/Nd_max": Nd_max,
                "theta/sigma_min": sigma_min, "theta/sigma_max": sigma_max,
                "theta/Bmax_min": Bmax_min, "theta/Bmax_max": Bmax_max,
            })

            for k in dict_avg_loss_recon_items:
                d_scalars[k] = dict_avg_loss_recon_items[k].result()
                dict_avg_loss_recon_items[k].reset_state()

            for k in dict_avg_loss_reg_p_items:
                d_scalars[k] = dict_avg_loss_reg_p_items[k].result()
                dict_avg_loss_reg_p_items[k].reset_state()

            for k in dict_avg_rmse_recon_items:
                d_scalars[k] = dict_avg_rmse_recon_items[k].result()
                dict_avg_rmse_recon_items[k].reset_state()

            for k in dict_avg_rmse_reg_p_items:
                d_scalars[k] = dict_avg_rmse_reg_p_items[k].result()
                dict_avg_rmse_reg_p_items[k].reset_state()

            d_scalars['lr_schedule'] = lr_schedule(step=step)

            export_summary_scalars(d_scalars, step=step, writer=summary_writer)

            logging.info(
                'Step %d: avg loss: %.3f, reconstruction loss: %.3f, parameter regression loss: %.3f.'
                % (step,
                float(d_scalars['loss_total']),
                float(d_scalars['loss_reconstruction']),
                float(d_scalars['loss_regress_params']))
            )

            avg_loss_total.reset_state()
            avg_loss_recon.reset_state()
            avg_loss_reg_p.reset_state()

            d_histograms = dict(
                [(f"enc_{v.name}", tf.identity(v)) for v in model_obj.encoder.trainable_variables] +
                [(f"dec_{v.name}", tf.identity(v)) for v in model_obj.decoder.trainable_variables]
            )
            export_summary_histograms(d_histograms, step=step, writer=summary_writer)

            save(save_manager, ckpt_number=step)

        # ------------------------------------------------
        # Long-term best model check
        # ------------------------------------------------
        if step % (10 * args.freq_log) == 0:

            current_long_loss = float(avg_loss_long_term.result().numpy())

            if current_long_loss < curr_best_loss:
                curr_best_loss = current_long_loss
                avg_loss_long_term.reset_state()

                logging.info(
                    'New long term best loss found: %.3f. Saving.' % curr_best_loss
                )

                save(save_manager_best, ckpt_number=step)


    # ------------------------------------------------
    # Final save after exiting loop
    # ------------------------------------------------
    save(save_manager, ckpt_number=step)

class Sampler:
    '''Class provides user friendly access to low-dimensional space for summary statistics analysis.'''
    def __init__(self, generator=None, iterator=None, **kwargs):
        '''Constructor builds NN model and loads its weights. In addition, a generator with true sun parameters is initialized.'''
        self.args = ExpSetup() # get experiment parameters from the ExpSetup class in this file. Make sure logdir is accurate.
        self.basename = kwargs.get('basename', 'model_best_ckpt')
        if 'logdir' in kwargs:
            self.args.logdir = kwargs.get('logdir')
        if not os.path.isdir(self.args.logdir):
            raise AssertionError('logdir %s from ExpSetup is not found. Quitting.' % self.args.logdir)
        self.check_hyper_params()
        self.prng = kwargs.get('prng', np.random.RandomState(1999))
        self.model_obj = None
        self.build_model()
        self.load_model(basename=self.basename)
        self.generator = generator
        self.iterator = iterator

    def sample(self, num_samples=10, return_noise_vectors=False, params=None):
        '''Function samples #num_samples observed vectors, then returns the mapped representation for each (size: [#num_samples, #stats]).
        if params is not None, existing generator is ignored and a new custom one is initialized.'''
        if params is None: # if none, fallback to default generator of the class instance
            if self.iterator is None:
                raise AssertionError('iterator is not defined. You must provide a parameter dict for the generator.')
            iterator = self.iterator 
        else:
            _, iterator = self.build_custom_generator(return_generator=True, **params)

        summary_space = np.zeros((num_samples, self.args.ndims_latent))
        ndarray_noise_timeseries = np.zeros((num_samples, self.args.len_timeseries, self.args.num_noise_channels))
        for i in range(num_samples):
            b = next(iterator) # sample contains a tuple of form (x, params, noise)
            x_i = np.expand_dims(b[0], axis=0).astype(np.float32) #creating minibatch size 1.
            noise_i = np.expand_dims(b[2], axis=0).astype(np.float32) #creating minibatch size 1.
            o_latent = self.model_obj.encoder(x_i, training=False).numpy()
            summary_space[i,...] = o_latent
            ndarray_noise_timeseries[i,...] = noise_i
        if return_noise_vectors:
            return summary_space, ndarray_noise_timeseries
        else:
            return summary_space

    def encode(self, samples):
        '''Function generates latent representations of the given observations (samples).
        samples should be of shape: [num_samples, len_timeseries, num_channels=1]'''
        o_latent = self.model_obj.encoder(samples)
        return o_latent

    def reconstruct(self, num_samples=10, params=None):
        '''Function samples #num_samples p_n vectors, then returns the reconstructions of the from their latent representation for each (size: [#num_samples, len_timeseries]
        if params is not None, existing generator is ignored and a new custom one is initialized.'''
        if params is None: # if none, fallback to default generator of the class instance
            if self.iterator is None:
                raise AssertionError('iterator is not defined. You must provide a parameter dict for the generator.')
            iterator = self.iterator 
        else:
            _, iterator = self.build_custom_generator(return_generator=True, **params)
        ndarray_timeseries = np.zeros((num_samples, self.args.len_timeseries))
        for i in range(num_samples):
            b = next(iterator) # sample contains a tuple of form (x, params, noise)
            x_i = np.expand_dims(b[0], axis=0).astype(np.float32) #creating minibatch size 1.
            noise_i = np.expand_dims(b[2], axis=0).astype(np.float32) #creating minibatch size 1.
            z_latent = self.model_obj.encoder(x_i, training=False)
            o_reconst = self.model_obj.decoder((z_latent, noise_i), training=False).numpy()
            ndarray_timeseries[i,...] = np.squeeze(o_reconst)
        return ndarray_timeseries

    def decode(self, tuple_summary_and_noise):
        '''Function reconstruct timeseries for a given tuple of (latent_representations, noise vectors).'''
        o_latent = tuple_summary_and_noise[0]
        noise_i = tuple_summary_and_noise[1]
        o_reconst = self.model_obj.decoder((o_latent, noise_i), training=False).numpy()
        return o_reconst

    def check_hyper_params(self):
        hp_manager = Manage_Hyper_Parameters(logdir=self.args.logdir)
        if hp_manager.args is None:
            raise AssertionError('Hyper-parameter configuration file %s is not found. Quitting.' % (hp_manager.param_config_fn))
        else:
            for attr in dir(self.args):
                if not attr.startswith('__') and attr not in ['logdir', 'batch_size']:  # don't get methods, ignore logdir and batch_size, as the optimization could have been done elsewhere.
                    if getattr(self.args, attr) != getattr(hp_manager.args, attr):
                        raise AssertionError('Mismatch of hyper-parameter attributes in the logdir %s and ExpSetup file in this script for attribute: %s' % (self.args.logdir, attr))

    def build_model(self):
        self.model_obj = Architecture(ndims_latent=self.args.ndims_latent, len_timeseries=self.args.len_timeseries, num_noise_channels=self.args.num_noise_channels)

    def load_model(self, basename='model_best_ckpt'):
        ckpt = tf.train.Checkpoint(encoder=self.model_obj.encoder, decoder=self.model_obj.decoder)
        save_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=self.args.logdir, max_to_keep=3, checkpoint_name=basename)
        # Restore model weights
        ckptname = get_ckptname(logdir=self.args.logdir, id=basename)
        try:
            ckpt.restore(ckptname).assert_existing_objects_matched().expect_partial()
        except: 
            raise AssertionError('Model weights with basename %s not found in logdir %s. Quitting.' % (basename, self.args.logdir))
        print('Model weights loaded from %s.' % ckptname)

    def build_custom_generator(self, return_generator=False, *, prng=None):
        if prng is None:
            prng = self.prng

        generator = src.generators.DataGenerator_SolarDynamo_SDDE_ENCA(
            prng=prng,
            Tobs=self.args.Tobs,
            saveat=self.args.saveat,
            num_noise_channels=self.args.num_noise_channels,
            Twarmup=self.args.Twarmup,
            dt=self.args.dt,
            tau_lims=self.args.tau_lims,
            T_lims=self.args.T_lims,
            Nd_lims=self.args.Nd_lims,
            sigma_lims=self.args.sigma_lims,
            Bmax_lims=self.args.Bmax_lims,
        )

        iterator = generator.__iter__()
        if return_generator:
            return generator, iterator
        else:
            self.generator = generator
            self.iterator = iterator

def get_ckptname(logdir, id):
    fl = glob.glob(os.path.join(logdir, '*')) # full paths of all files
    fn = [it for it in fl if id in os.path.basename(it)] #keep only files that match id in the basenames
    flb = [os.path.basename(it) for it in fn] #keep only basenames of matching abs path filenames
    ind = np.argmax([int(it[len(id)+1:].split('.')[0]) for it in flb]) #get highest ckpt ind of matching filenames
    fname = fn[ind] #get abs filename of the matching index file
    fname = '.'.join(fname.split('.')[:-1]) # truncate extension of the file: e.g., /path/to/file/{id}-{ind}.{extension} -> /path/to/file/{id}-{ind}
    return fname

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs visible to TensorFlow:", gpus)
    main()
    
