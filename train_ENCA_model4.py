#Author: Simone Ulzega, February 2026, simone.ulzega@zhaw.ch
################
# Conditional ENCA training pipeline for a model M(theta, pA) -> pB.
# The encoder maps pB to latent summaries, and the decoder reconstructs pB from
# those latent summaries plus the same pA that was used to generate pB. The first
# P latent variables are supervised to regress the P model parameters; extra
# latent dimensions are allowed and remain free.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

##################################################################################################
def build_generator(args):
    """
    Hook for the model-specific generator.

    Expected generator contract:
      yield pB, theta, pA

    where:
      pB    has shape [len_timeseries, num_output_channels]
      theta has shape [num_model_parameters]
      pA    has shape [len_timeseries, num_condition_channels]
    """
    raise NotImplementedError(
        "Model generator is not wired yet. Replace build_generator(args) with a "
        "generator that samples theta and pA, computes pB = M(theta, pA), and "
        "yields (pB, theta, pA)."
    )

##################################################################################################
class Architecture:
    '''Use this class to customize the conditional ENCA architecture.'''
    def __init__(self, ndims_latent, len_timeseries, num_output_channels, num_condition_channels):
        self.ndims_latent = ndims_latent
        self.len_timeseries = len_timeseries
        self.num_output_channels = num_output_channels
        self.num_condition_channels = num_condition_channels

        self.encoder = self.encoder_fn()
        self.decoder = self.decoder_fn()

    def encoder_fn(self):
        '''pB_input size: [bs, #len_timeseries, #num_output_channels]
        Implements encoder of only convolutional and maxpooling operators'''
        conv_fn = lambda filters, act=None, name=None: tf.keras.layers.Conv1D(filters=filters, kernel_size=3, activation=act, name=name)
        pB_input = tf.keras.layers.Input(shape=[self.len_timeseries, self.num_output_channels], name='pB_observation')
        x = pB_input
        self.num_conv_filters = [[16, 16], [32, 32]]
        for i in range(len(self.num_conv_filters)):
            if i != 0:
                x = tf.keras.layers.MaxPool1D(pool_size=2, name='maxpool%d'%(i+1))(x)
            for j in range(len(self.num_conv_filters[i])):
                x = conv_fn(filters=self.num_conv_filters[i][j], act='relu', name='conv%d_%d'%((i+1), (j+1)))(x) #[batch_size, len_timeseries, num_conv_filters[-1]]
        x = conv_fn(filters=self.ndims_latent, act=None, name='final_conv')(x)  # [batch_size, len_timeseries, ndims_latent]
        latent_space = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        return tf.keras.Model(inputs=pB_input, outputs=latent_space)

    def decoder_fn(self):
        '''latent_mappings size: [bs, #ndims_latent]
        pA_condition size: [bs, #len_timeseries, #num_condition_channels]
        output: [bs, #len_timeseries, #num_output_channels]'''
        latent_mappings = tf.keras.layers.Input(shape=[self.ndims_latent], name='latent_representations')
        pA_condition = tf.keras.layers.Input(shape=[self.len_timeseries, self.num_condition_channels], name='pA_condition')
        tile_ldims_layer = tf.keras.layers.Lambda(function=lambda x: tf.tile(tf.expand_dims(x, axis=1), multiples=[1, self.len_timeseries, 1]), name='tile_latent_space') 
        x = tf.keras.layers.Concatenate(axis=-1, name='concatenate_pA_and_latent_dims')([tile_ldims_layer(latent_mappings), pA_condition])
        num_units = 16
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_units, return_sequences=True, dtype=tf.float32, name='lstm_cell_1'), name='Bi-cell-1')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_units, return_sequences=True, dtype=tf.float32, name='lstm_cell_2'), name='Bi-cell-2')(x)
        x = tf.keras.layers.Dense(units=self.num_output_channels, activation=None, name='pred')(x)
        x = tf.keras.layers.Reshape([self.len_timeseries, self.num_output_channels], name='output_shape')(x)
        return tf.keras.Model(inputs=(latent_mappings, pA_condition), outputs=x)

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
        tag = "conditional_smoke"   # change this when you test something new
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_logdir = os.path.join(
            os.getcwd(),
            "ode_ENCA_runs",
            f"{run_id}_{tag}"
        )
        self.logdir = os.environ.get("ENCA_LOGDIR", default_logdir)

        self.ndims_latent = 6
        self.num_output_channels = 1      # pB channels
        self.num_condition_channels = 1   # pA channels

        # Replace these placeholders with the model parameters.
        self.parameter_names = ("theta1", "theta2", "theta3", "theta4", "theta5", "theta6")
        self.parameter_lims = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        )
        self.num_model_parameters = len(self.parameter_names)
        if len(self.parameter_lims) != self.num_model_parameters:
            raise ValueError("parameter_lims must have the same length as parameter_names.")
        if self.ndims_latent < self.num_model_parameters:
            raise ValueError(
                f"ndims_latent={self.ndims_latent} is smaller than "
                f"num_model_parameters={self.num_model_parameters}. "
                "Conditional ENCA needs at least one latent dimension per regressed parameter."
            )

        # Observation settings. Adjust to the model once available.
        self.Tobs = 100.0
        self.saveat = 1.0

        # derived length used by NN + dataset shapes
        ratio = self.Tobs / self.saveat
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"Tobs ({self.Tobs}) must be divisible by saveat ({self.saveat}).")
        self.len_timeseries = int(round(ratio))  # equals Tobs if saveat=1.0

        self.batch_size = 300
        self.max_training_steps = int(2e6) # int(3000) # int(3e6)
        self.freq_log = 500

        # Loss setup
        # "legacy_chisq" reproduces the original implementation.
        # "balanced_mse" uses normalized MSEs with comparable reductions.
        self.loss_mode = "balanced_mse"
        self.lambda_recon = 1.0
        self.lambda_reg = 1.0
        self.recon_scale_eps = 1e-3
        
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
    # Define a generator function for pB observations, parameters, and pA conditions.
    gen_train = build_generator(args)

    def validate_batch_shapes(pB, params, pA):
        if pB.shape.rank != 3:
            raise ValueError(f"pB must have shape [batch, len_timeseries, channels], got {pB.shape}.")
        if params.shape.rank != 2:
            raise ValueError(f"params must have shape [batch, num_model_parameters], got {params.shape}.")
        if pA.shape.rank != 3:
            raise ValueError(f"pA must have shape [batch, len_timeseries, channels], got {pA.shape}.")
        if pB.shape[1] != args.len_timeseries:
            raise ValueError(f"pB has len_timeseries={pB.shape[1]}, expected {args.len_timeseries}.")
        if pB.shape[2] != args.num_output_channels:
            raise ValueError(f"pB has num_output_channels={pB.shape[2]}, expected {args.num_output_channels}.")
        if params.shape[1] != args.num_model_parameters:
            raise ValueError(
                f"params has {params.shape[1]} columns, expected "
                f"num_model_parameters={args.num_model_parameters}."
            )
        if pA.shape[1] != args.len_timeseries:
            raise ValueError(f"pA has len_timeseries={pA.shape[1]}, expected {args.len_timeseries}.")
        if pA.shape[2] != args.num_condition_channels:
            raise ValueError(f"pA has num_condition_channels={pA.shape[2]}, expected {args.num_condition_channels}.")
    
    def next_batch_from_generator(gen, batch_size):
        """Collect batch_size samples from the Python generator (main thread)."""
        pBs = []
        ps = []
        pAs = []
        for _ in range(batch_size):
            pB0, p0, pA0 = next(gen)   # gen yields numpy arrays
            pBs.append(pB0)
            ps.append(p0)
            pAs.append(pA0)

        # Stack into numpy batches
        pB_np = np.stack(pBs, axis=0).astype(np.float32)    # (B, len, pB channels)
        p_np = np.stack(ps, axis=0).astype(np.float32)      # (B, P)
        pA_np = np.stack(pAs, axis=0).astype(np.float32)    # (B, len, pA channels)

        # Convert to TF tensors
        pB = tf.convert_to_tensor(pB_np)
        params = tf.convert_to_tensor(p_np)
        pA = tf.convert_to_tensor(pA_np)
        return pB, params, pA

    ##################################################################################################
    # Define the architecture
    model_obj = Architecture(
        ndims_latent=args.ndims_latent,
        len_timeseries=args.len_timeseries,
        num_output_channels=args.num_output_channels,
        num_condition_channels=args.num_condition_channels,
    )
    model_obj.encoder.summary()
    model_obj.decoder.summary()

    ##################################################################################################
    # Define objective functions
    class ChiSquareStatistic(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            '''Implements sum((y_true - y_pred) ** 2 / y_true).'''
            sd = tf.math.squared_difference(y_true, y_pred)
            return tf.math.reduce_sum(sd / tf.math.maximum(tf.math.pow(y_true, 2), 1e-6), axis= -1)

    param_widths = tf.constant(
        [hi - lo for lo, hi in args.parameter_lims],
        dtype=tf.float32,
    )

    @tf.function
    def loss_reconstruction_fn_legacy(x, x_pred, return_each_dim=False):
        if return_each_dim:
            d = {}
            for i in range(x.shape[-1]): # do over all channels of observation (it's just 1 in our example.)
                # d['MSE_pB_ch_%d'%(i+1)] = tf.keras.losses.MeanSquaredError(name='MSE')(x[...,i],x_pred[...,i]) # default behavior is sum over batch size: (average over minibatch size)
                d['ChiSquare_pB_ch_%d'%(i+1)] = ChiSquareStatistic(name='ChiSquare')(y_true=x[...,i], y_pred=x_pred[...,i]) / args.len_timeseries # take average of ChiSquare across the timeseries elements (as to scale similarly with ChiSquare of latent parameters)
            return d
        else:
            # return tf.keras.losses.MeanSquaredError(name='MSE')(x,x_pred)
            return ChiSquareStatistic(name='ChiSquare')(x,x_pred) / args.len_timeseries

    @tf.function
    def loss_regress_params_fn_legacy(params, params_pred, return_each_dim=False):
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

    @tf.function
    def loss_reconstruction_fn_balanced(x, x_pred, return_each_dim=False):
        """
        Per-sample normalized MSE:
          1. compute one RMS amplitude scale per sample and channel
          2. normalize reconstruction error by that scale
          3. average over time, then over batch
        """
        rms_scale = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True))
        rms_scale = tf.maximum(rms_scale, tf.constant(args.recon_scale_eps, dtype=x.dtype))
        sq_error = tf.square((x - x_pred) / rms_scale)
        per_sample_channel = tf.reduce_mean(sq_error, axis=1)  # [batch, channels]

        if return_each_dim:
            d = {}
            for i in range(x.shape[-1]):
                d[f'NormMSE_pB_ch_{i+1}'] = tf.reduce_mean(per_sample_channel[:, i])
            return d

        return tf.reduce_mean(per_sample_channel)

    @tf.function
    def loss_regress_params_fn_balanced(params, params_pred, return_each_dim=False):
        """
        Mean squared error after normalizing each parameter by its prior width.
        Only the first num_model_parameters latent dimensions are supervised.
        """
        num_params = params._shape_as_list()[-1]
        params_pred_used = params_pred[..., :num_params]
        widths = tf.cast(param_widths[:num_params], params.dtype)
        sq_error = tf.square((params - params_pred_used) / widths)

        if return_each_dim:
            d = {}
            for i in range(num_params):
                d[f'NormMSE_z_{i+1}'] = tf.reduce_mean(sq_error[:, i])
            return d

        return tf.reduce_mean(sq_error)
    # Define additional metrics for logging at every

    ##################################################################################################
    # Define optimizer 
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
    def train_step(model, pB, params, pA, optimizer):
        """Single training step (no logging inside)."""
        with tf.GradientTape(persistent=False) as tape:
            z_latent = model.encoder(pB, training=True)
            pB_reconst = model.decoder((z_latent, pA), training=True)

            if args.loss_mode == "legacy_chisq":
                dict_reconstruction = loss_reconstruction_fn_legacy(pB, pB_reconst, return_each_dim=True)
                loss_reconstruction = tf.reduce_sum(list(dict_reconstruction.values()))

                dict_regression = loss_regress_params_fn_legacy(params, z_latent, return_each_dim=True)
                loss_regress_params = tf.reduce_sum(list(dict_regression.values()))
            elif args.loss_mode == "balanced_mse":
                dict_reconstruction = loss_reconstruction_fn_balanced(pB, pB_reconst, return_each_dim=True)
                loss_reconstruction = tf.reduce_mean(tf.stack(list(dict_reconstruction.values())))

                dict_regression = loss_regress_params_fn_balanced(params, z_latent, return_each_dim=True)
                loss_regress_params = tf.reduce_mean(tf.stack(list(dict_regression.values())))
            else:
                raise ValueError(f"Unknown loss_mode: {args.loss_mode}")

            loss = args.lambda_recon * loss_reconstruction + args.lambda_reg * loss_regress_params

        trainable_variables = model.encoder.trainable_variables + model.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        if global_gradient_clipnorm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clipnorm)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return (loss_reconstruction, loss_regress_params, loss), (z_latent, pB_reconst), (dict_reconstruction, dict_regression)
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
    last_step_saved = False
    
    gen_iter = iter(gen_train)
    
    print("Building first batch...")
    t0 = time.time()
    pB, params, pA = next_batch_from_generator(gen_iter, args.batch_size)
    validate_batch_shapes(pB, params, pA)
    print(f"First batch built in {time.time()-t0:.1f}s: pB={pB.shape}, params={params.shape}, pA={pA.shape}")

    # --- quick finiteness + stats checks (on first batch) ---
    def assert_finite(name, t):
        t_np = t.numpy()
        if not np.isfinite(t_np).all():
            bad = np.where(~np.isfinite(t_np))
            raise ValueError(f"{name} has non-finite values at {bad[:3]}")

    assert_finite("pB", pB)
    assert_finite("params", params)
    assert_finite("pA", pA)
    print("Batch finiteness: OK")

    print("pB stats:", float(tf.reduce_min(pB)), float(tf.reduce_max(pB)), float(tf.reduce_mean(pB)))
    print("params stats:", float(tf.reduce_min(params)), float(tf.reduce_max(params)), float(tf.reduce_mean(params)))
    print("pA stats:", float(tf.reduce_min(pA)), float(tf.reduce_max(pA)), float(tf.reduce_mean(pA)))
    print(f"loss mode: {args.loss_mode} (lambda_recon={args.lambda_recon}, lambda_reg={args.lambda_reg})")

    # --- warm up tf.function tracing/compilation ---
    print("Warming up train_step (tf.function tracing/compilation)...")
    t1 = time.time()
    _ = train_step(model=model_obj, pB=pB, params=params, pA=pA, optimizer=optimizer)
    print(f"Warmup train_step done in {time.time()-t1:.2f}s")

    # ---------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------
    while True:
        # 1) build batch (python)
        pB, params, pA = next_batch_from_generator(gen_iter, args.batch_size)

        # 2) train step (tf.function)
        loss_tuple, z_and_pB, dict_mse = train_step(
            model=model_obj, pB=pB, params=params, pA=pA, optimizer=optimizer
        )

        # unpack
        z_latent, pB_reconst = z_and_pB
        loss_reconstruction, loss_regress_params, loss_total = loss_tuple
        dict_rec, dict_reg = dict_mse

        # 3) NaN/Inf guard
        tf.debugging.assert_all_finite(loss_total, "loss_total has NaN or Inf")

        # 4) step / stopping
        step = int(optimizer.iterations.numpy())
        reached_max_steps = step >= args.max_training_steps

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
        for i_ in range(pB.shape[-1]):
            k = f'RMSE_pB_ch_{i_+1}'
            if k not in dict_avg_rmse_recon_items:
                dict_avg_rmse_recon_items[k] = tf.keras.metrics.RootMeanSquaredError(
                    name='rmse_reconstruction', dtype=tf.float32
                )
            dict_avg_rmse_recon_items[k].update_state(
                y_true=pB[..., i_], y_pred=pB_reconst[..., i_]
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
        if (step % args.freq_log == 0) or reached_max_steps:

            d_scalars = {
                'loss_total': avg_loss_total.result(),
                'loss_reconstruction': avg_loss_recon.result(),
                'loss_regress_params': avg_loss_reg_p.result(),
                'loss_weight/lambda_recon': tf.constant(args.lambda_recon, dtype=tf.float32),
                'loss_weight/lambda_reg': tf.constant(args.lambda_reg, dtype=tf.float32),
            }

            # --- Parameter ranges in current batch ---
            # (safe even with batch_size=1; then min==max)
            p = params.numpy()   # shape (B, P)
            range_parts = []
            for i_, name in enumerate(args.parameter_names):
                p_min = float(p[:, i_].min())
                p_max = float(p[:, i_].max())
                range_parts.append(f"{name}[{p_min:.3f},{p_max:.3f}]")
                d_scalars[f"theta/{name}_min"] = p_min
                d_scalars[f"theta/{name}_max"] = p_max

            logging.info("params ranges: " + " ".join(range_parts))

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
            last_step_saved = True

        # ------------------------------------------------
        # Long-term best model check
        # ------------------------------------------------
        if (step % (10 * args.freq_log) == 0) or reached_max_steps:

            current_long_loss = float(avg_loss_long_term.result().numpy())

            if current_long_loss < curr_best_loss:
                curr_best_loss = current_long_loss
                avg_loss_long_term.reset_state()

                logging.info(
                    'New long term best loss found: %.3f. Saving.' % curr_best_loss
                )

                save(save_manager_best, ckpt_number=step)

        if reached_max_steps:
            logging.info("Training is completed.")
            break


    # ------------------------------------------------
    # Final save after exiting loop
    # ------------------------------------------------
    if not last_step_saved:
        save(save_manager, ckpt_number=step)

class Sampler:
    '''Class provides user friendly access to low-dimensional space for summary statistics analysis.'''
    def __init__(self, generator=None, iterator=None, **kwargs):
        '''Constructor builds NN model and loads its weights.'''
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

    def sample(self, num_samples=10, params=None):
        '''Function samples #num_samples pB vectors, then returns their mapped representation (size: [#num_samples, #stats]).
        if params is not None, existing generator is ignored and a new custom one is initialized.'''
        if params is None: # if none, fallback to default generator of the class instance
            if self.iterator is None:
                raise AssertionError('iterator is not defined. You must provide a parameter dict for the generator.')
            iterator = self.iterator 
        else:
            _, iterator = self.build_custom_generator(return_generator=True, **params)

        summary_space = np.zeros((num_samples, self.args.ndims_latent))
        for i in range(num_samples):
            b = next(iterator) # sample contains a tuple of form (pB, params, pA)
            pB_i = np.expand_dims(b[0], axis=0).astype(np.float32) #creating minibatch size 1.
            o_latent = self.model_obj.encoder(pB_i, training=False).numpy()
            summary_space[i,...] = o_latent
        return summary_space

    def encode(self, samples):
        '''Function generates latent representations of the given pB observations (samples).
        samples should be of shape: [num_samples, len_timeseries, num_channels=1]'''
        o_latent = self.model_obj.encoder(samples)
        return o_latent

    def reconstruct(self, num_samples=10, params=None):
        '''Function samples #num_samples pB vectors, then returns reconstructions from their latent representation and pA condition.
        if params is not None, existing generator is ignored and a new custom one is initialized.'''
        if params is None: # if none, fallback to default generator of the class instance
            if self.iterator is None:
                raise AssertionError('iterator is not defined. You must provide a parameter dict for the generator.')
            iterator = self.iterator 
        else:
            _, iterator = self.build_custom_generator(return_generator=True, **params)
        ndarray_timeseries = np.zeros((num_samples, self.args.len_timeseries))
        for i in range(num_samples):
            b = next(iterator) # sample contains a tuple of form (pB, params, pA)
            pB_i = np.expand_dims(b[0], axis=0).astype(np.float32) #creating minibatch size 1.
            pA_i = np.expand_dims(b[2], axis=0).astype(np.float32) #creating minibatch size 1.
            z_latent = self.model_obj.encoder(pB_i, training=False)
            o_reconst = self.model_obj.decoder((z_latent, pA_i), training=False).numpy()
            ndarray_timeseries[i,...] = np.squeeze(o_reconst)
        return ndarray_timeseries

    def decode(self, latent_representations, pA_condition):
        '''Function reconstructs pB for given latent_representations and pA_condition.'''
        o_reconst = self.model_obj.decoder((latent_representations, pA_condition), training=False).numpy()
        return o_reconst

    def check_hyper_params(self):
        hp_manager = Manage_Hyper_Parameters(logdir=self.args.logdir)
        if hp_manager.args is None:
            raise AssertionError('Hyper-parameter configuration file %s is not found. Quitting.' % (hp_manager.param_config_fn))
        else:
            for attr in dir(self.args):
                if not attr.startswith('__') and attr not in ['logdir', 'batch_size']:  # don't get methods, ignore logdir and batch_size, as the optimization could have been done elsewhere.
                    if not hasattr(hp_manager.args, attr):
                        continue
                    if getattr(self.args, attr) != getattr(hp_manager.args, attr):
                        raise AssertionError('Mismatch of hyper-parameter attributes in the logdir %s and ExpSetup file in this script for attribute: %s' % (self.args.logdir, attr))

    def build_model(self):
        self.model_obj = Architecture(
            ndims_latent=self.args.ndims_latent,
            len_timeseries=self.args.len_timeseries,
            num_output_channels=self.args.num_output_channels,
            num_condition_channels=self.args.num_condition_channels,
        )

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

    def build_custom_generator(self, return_generator=False, **kwargs):
        generator = build_generator(self.args)

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
    
