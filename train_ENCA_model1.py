#Author: Firat Ozdemir, August 2020, firat.ozdemir@datascience.ch
################
# This training pipeline has an encoder-decoder-like architecture. From input timeseries to sufficient statistics space, then
# using the noise vectors back to the reconstruction of the initial input signal.


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
assert tf.__version__[0] == '2' # this script is intended for tf v2.2+
import os, glob
import shutil
import datetime
import numpy as np
import sys
import json
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

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
        if self.args is None:
            return None
        save_args = False
        # for k in args:
        for k in dir(args):
            if k.startswith('__'):
                continue
            if not hasattr(self.args, k):
                print('WARNING: key "%s" was missing in the new hyper-parameter configuration; will renew file.' % k)
                save_args = True
            else:
                if getattr(self.args, k) != getattr(args, k):
                    print('ERROR! Mismatch in hyper-parameter settings file. Parameter %s was %s, now it is %s. Quitting training.' % (k, getattr(self.args, k), getattr(args, k)))
        if save_args:
            self.save_parameters(args)
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
    '''Use this class to define the hyper parameters to be used for the AE.'''
    def __init__(self):
        self.logdir = '/tmp/model1_ENCA'
        self.ndims_latent = 3 # #summary stats
        self.num_noise_channels = 1 # #different noise vectors
        self.num_model_parameters = 2 # NLAR1 model has c and sigma as model parameters.
        self.len_timeseries = 200
        self.batch_size = 300
        self.max_training_steps = int(3*1e6) # maximum number of training steps unless optimization gets killed.
        self.freq_log = 100 # frequency to update logged values in tensorboard.

##################################################################################################
def main(args):

    args = ExpSetup()

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    ##################################################################################################
    # Define a generator function for observations, parameters, and noise vectors
    x_0 = 0.25
    c_lims = [4.2, 5.8]
    sigma_lims = [0.005, 0.025]
    #############################################
    fn = lambda c, x_old, sigma, epsilon: c * x_old**2 * (1-x_old) + sigma * epsilon
    #############################################
    gen_train = src.generators.DataGenerator_NLAR1_Simplified(len_timeseries=args.len_timeseries, x_0=x_0, c_lims=c_lims, sigma_lims=sigma_lims, fn=fn)
    dataset_train = tf.data.Dataset.from_generator(lambda: gen_train, output_types=(tf.float32, tf.float32, tf.float32))
    dataset_train = dataset_train.repeat(count=1)
    dataset_train = dataset_train.batch(args.batch_size, drop_remainder=True)
    dataset_train = dataset_train.prefetch(buffer_size=10) # #number of minibatches to pre-fetch.

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
    # lr_schedule = src.utils_tf.LearningRateScheduleExponentialDecayWithLinearWarmup(steps_warmup=args.linear_warmup_steps, initial_learning_rate=1.e-3, decay_steps=int(6*1e3), decay_rate=0.92, staircase=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.e-3, decay_steps=int(6*1e3), decay_rate=0.92, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #, clipnorm=1e5, clipvalue=1.) #clips |gradients| above clipvalue to prevent exploding., #clipnorm: preventative measure for divergence. Unfortunately both are clipping individually for each gradient, which can change the direction of the gradients..
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
    
    @tf.function
    def export_summary_scalars(dict_name_and_val, step, writer):
        '''dict_name_and_val = {'name1': value1, 'name2': value2, ...}'''
        if not isinstance(dict_name_and_val, dict):
            raise AssertionError('dict_name_and_val must be a dictionary.')
        with writer.as_default():
            for k in dict_name_and_val.keys():
                tf.summary.scalar(k, dict_name_and_val[k], step=step)

    @tf.function
    def export_summary_histograms(dict_name_and_val, step, writer):
        if not isinstance(dict_name_and_val, dict):
            raise AssertionError('dict_name_and_val must be a dictionary.')
        with writer.as_default():
            for k in dict_name_and_val.keys():
                tf.summary.histogram(k, dict_name_and_val[k], step=step)

    ##################################################################################################
    # wrap a training step for performance gains.
    @tf.function
    def train_step(model, x, params, noise, optimizer):
        '''Single training step.'''
        with tf.GradientTape(persistent=False) as tape: #persistent=True if .gradient() will be called multiple times (e.g., multiple losses)
            z_latent = model.encoder(x, training=True)
            x_reconst = model.decoder((z_latent, noise), training=True)
            dict_reconstruction_mse = loss_reconstruction_fn(x, x_reconst, return_each_dim=True)
            loss_reconstruction = tf.math.reduce_sum(list(dict_reconstruction_mse.values()))
            dict_regress_params_mse = loss_regress_params_fn(params, z_latent, return_each_dim=True)
            loss_regress_params = tf.math.reduce_sum(list(dict_regress_params_mse.values()))
            loss = loss_reconstruction + loss_regress_params
            trainable_variables = model.encoder.trainable_variables + model.decoder.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            if global_gradient_clipnorm is not None:
                gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clipnorm, name='clip_gradients_by_global_norm')
            if tf.equal(optimizer.iterations % args.freq_log, 0):
                l_avg_grads= [tf.math.reduce_mean(g) for g in gradients]
                l_absmax_grads= [tf.math.reduce_max(tf.math.abs(g)) for g in gradients]
                l_gradmean_names = ['gradient_mean_'+v.name for v in trainable_variables]
                l_gradabsmax_names = ['gradient_absmax_'+v.name for v in trainable_variables]
                # export avg and absolute max gradients of variables to tensorboard
                d_grads = {**dict(zip(l_gradmean_names, l_avg_grads)), **dict(zip(l_gradabsmax_names, l_absmax_grads))}
                export_summary_scalars(dict_name_and_val=d_grads, step=optimizer.iterations, writer=summary_writer)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
        return (loss_reconstruction, loss_regress_params), (z_latent, x_reconst), (dict_reconstruction_mse, dict_regress_params_mse)

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
    for x, params, noise in dataset_train:
        num_step = optimizer.iterations
        if num_step >= args.max_training_steps:
            logging.info('Training is completed.')
            break
        loss_tuple, z_and_x, dict_mse = train_step(model=model_obj, x=x, params=params, noise=noise, optimizer=optimizer)
        z_latent, x_reconst = z_and_x
        loss_reconstruction, loss_regress_params = loss_tuple
        loss_total = loss_reconstruction + loss_regress_params
        dict_rec, dict_reg = dict_mse
        # Update loggers for tensorboard
        avg_loss_recon.update_state(loss_reconstruction)
        avg_loss_reg_p.update_state(loss_regress_params)
        avg_loss_total.update_state(loss_total) # aggregate values since last flush
        avg_loss_long_term.update_state(loss_total)
        # A little patchy way to keep track of each reconstructed signal and noise channel
        for k in dict_rec:
            if k not in dict_avg_loss_recon_items:
                dict_avg_loss_recon_items[k] = tf.keras.metrics.Mean(name=k, dtype=tf.float32)
            dict_avg_loss_recon_items[k].update_state(dict_rec[k])                
        for k in dict_reg:
            if k not in dict_avg_loss_reg_p_items:
                dict_avg_loss_reg_p_items[k] = tf.keras.metrics.Mean(name=k, dtype=tf.float32)
            dict_avg_loss_reg_p_items[k].update_state(dict_reg[k])
        
         # Record RMSE for reconstruction and regressed parameters regardless of the loss
        for i_ in range(x.shape[-1]):
            k = 'RMSE_x_ch_%d'%(i_+1)
            if k not in dict_avg_rmse_recon_items:
                dict_avg_rmse_recon_items[k] = tf.keras.metrics.RootMeanSquaredError(name='rmse_reconstruction', dtype=tf.float32)
            dict_avg_rmse_recon_items[k].update_state(y_true=x[...,i_], y_pred=x_reconst[...,i_])
        for i_ in range(params.shape[-1]):
            k = 'RMSE_z_ch_%d'%(i_+1)
            if k not in dict_avg_rmse_reg_p_items:
                dict_avg_rmse_reg_p_items[k] = tf.keras.metrics.RootMeanSquaredError(name='rmse_regularization', dtype=tf.float32)
            dict_avg_rmse_reg_p_items[k].update_state(y_true=params[...,i_], y_pred=z_latent[...,i_])
        # Export status to tensorboard
        if tf.equal(optimizer.iterations % args.freq_log, 0):
            d_scalars = {'loss_total': avg_loss_total.result(), 'loss_reconstruction': avg_loss_recon.result(), 'loss_regress_params': avg_loss_reg_p.result()}
            for k in dict_avg_loss_recon_items:
                d_scalars[k] = dict_avg_loss_recon_items[k].result()
                dict_avg_loss_recon_items[k].reset_states()
            for k in dict_avg_loss_reg_p_items:
                d_scalars[k] = dict_avg_loss_reg_p_items[k].result()
                dict_avg_loss_reg_p_items[k].reset_states()
            for k in dict_avg_rmse_recon_items:
                d_scalars[k] = dict_avg_rmse_recon_items[k].result()
                dict_avg_rmse_recon_items[k].reset_states()
            for k in dict_avg_rmse_reg_p_items:
                d_scalars[k] = dict_avg_rmse_reg_p_items[k].result()
                dict_avg_rmse_reg_p_items[k].reset_states()
            d_scalars['lr_schedule'] = lr_schedule(step=num_step)
            export_summary_scalars(dict_name_and_val=d_scalars, step=optimizer.iterations, writer=summary_writer)
            # Print current loss status to terminal
            logging.info('Step %d: avg loss: %.3f, reconstruction loss: %.3f, parameter regression loss: %.3f.' % \
                    (num_step, d_scalars['loss_total'], d_scalars['loss_reconstruction'], d_scalars['loss_regress_params']))
            avg_loss_total.reset_states() #reset kept history of loss
            avg_loss_recon.reset_states()
            avg_loss_reg_p.reset_states()
            # Export trainable variables to tboard histogram
            # TODO: consider speeding this up.
            l_enc = list(zip(*[['enc_'+v.name, v.value()] for v in model_obj.encoder.trainable_variables]))
            l_dec = list(zip(*[['dec_'+v.name, v.value()] for v in model_obj.decoder.trainable_variables]))
            d_histograms = {**dict(zip(l_enc[0], l_enc[1])), **dict(zip(l_dec[0], l_dec[1]))}
            export_summary_histograms(dict_name_and_val=d_histograms, step=optimizer.iterations, writer=summary_writer)
            # Save the current state of the model weights on disk.
            save(save_manager, ckpt_number=optimizer.iterations)
        # Check loss for long term best reconstruction
        if tf.equal(optimizer.iterations % (10 * args.freq_log), 0):
            if avg_loss_long_term.result() < curr_best_loss:
                curr_best_loss = avg_loss_long_term.result()
                avg_loss_long_term.reset_states()
                logging.info('New long term best loss found: %.3f. Saving.' % curr_best_loss)
                save(save_manager_best, ckpt_number=optimizer.iterations)
            
    ##################################################################################################
    # Save model once again on the exit
    save(save_manager, ckpt_number=optimizer.iterations)

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

    def sample(self, num_samples=10, return_noise_vectors=False, return_model_parameters=False, return_observations=False, params=None):
        '''Function samples #num_samples observed vectors, then returns the mapped representation for each (size: [#num_samples, #stats]).
        if params is not None, existing generator is ignored and a new custom one is initialized.'''
        if params is None: # if none, fallback to default generator of the class instance
            if self.iterator is None:
                raise AssertionError('iterator is not defined. You must provide a parameter dict for the generator.')
            iterator = self.iterator 
        else:
            _, iterator = self.build_custom_generator(return_generator=True, **params)

        summary_space = np.zeros((num_samples, self.args.ndims_latent))
        model_params = np.zeros((num_samples, self.args.num_model_parameters))
        ndarray_noise_timeseries = np.zeros((num_samples, self.args.len_timeseries, self.args.num_noise_channels))
        observations = np.zeros((num_samples, self.args.len_timeseries))
        for i in range(num_samples):
            b = next(iterator) # sample contains a tuple of form (x, params, noise)
            x_i = np.expand_dims(b[0], axis=0).astype(np.float32) #creating minibatch size 1.
            noise_i = np.expand_dims(b[2], axis=0).astype(np.float32) #creating minibatch size 1.
            o_latent = self.model_obj.encoder(x_i, training=False).numpy()
            summary_space[i,...] = o_latent
            ndarray_noise_timeseries[i,...] = noise_i
            model_params[i,...] = b[1]
            observations[i,...] = np.squeeze(b[0])
        elms_return = [summary_space]
        if return_noise_vectors:
            elms_return += [ndarray_noise_timeseries]
        if return_model_parameters:
            elms_return += [model_params]
        if return_observations:
            elms_return += [observations]
        if len(elms_return) == 1:
            return elms_return[0]
        else:
            return elms_return

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
        
    def build_custom_generator(self, return_generator=False, **kwargs):
        '''See inside DataGenerator_SolarDynamo for the parameters one can modify for custom generator.'''
        if 'prng' in kwargs:
            prng = kwargs.pop('prng')
        else:
            prng = self.prng
        x_0 = kwargs.pop('x_0', 0.25)
        c_lims = kwargs.pop('c_lims', [4.2, 5.8])
        sigma_lims = kwargs.pop('sigma_lims', [0.005, 0.025])
        #############################################
        fn = lambda c, x_old, sigma, epsilon: c * x_old**2 * (1-x_old) + sigma * epsilon
        #############################################
        func = kwargs.pop('fn', fn)
        generator = src.generators.DataGenerator_NLAR1_Simplified(len_timeseries=self.args.len_timeseries, prng=prng, x_0=x_0, c_lims=c_lims, sigma_lims=sigma_lims, fn=func, **kwargs)
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
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices([], 'GPU') # do not use any GPU
    # tf.config.set_visible_devices(gpus[0], 'GPU') ## Use only GPU: 0
    main(0)
