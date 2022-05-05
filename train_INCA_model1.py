#Author: Firat Ozdemir, February 2021, firat.ozdemir@datascience.ch
################
# This training pipeline has an encoder-aggregator-like architecture. From a vector of input timeseries to a vector sufficient statistics space, then
# using the free parameters in summary stats space, find a weighting vector to average corresponding regressed parameters in statistics space.


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
    This example uses a fully convolutional encoder and a FC aggregator.'''
    def __init__(self, mb_of_same_realizations, ndims_latent, ndims_free_latent_parameters, len_timeseries, num_noise_channels, ndims_output):
        self.ndims_latent = ndims_latent
        self.ndims_free_latent_parameters = ndims_free_latent_parameters
        self.len_timeseries = len_timeseries
        self.mb_of_same_realizations = mb_of_same_realizations
        self.num_input_channels = 1 #assuming a given observed timeseries is a single channel (vector)
        self.num_noise_channels = num_noise_channels
        self.ndims_output = ndims_output

        self.encoder = self.encoder_fn()
        self.aggregator = self.aggregator_fn(ndims_free_latent_parameters)

    def encoder_fn(self):
        '''x_input size: [bs, #len_timeseries, #num_input_channels] 
        Implements encoder of only convolutional and maxpooling operators'''
        def maxpool1D_nd_fn(x):
            '''First reshape input into 3 dims, apply maxpool1d, reshape back to ndims'''
            shp = x.shape
            x = tf.reshape(x, shape=[-1, shp[-2], shp[-1]])
            x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
            x = tf.reshape(x, shape=[-1] + [s for s in shp[1:-2]] + [x.shape[-2], shp[-1]])
            return x
        
        class Conv1D_ND(tf.keras.layers.Layer):
            def __init__(self, filters, act='relu', **kwargs):
                super(Conv1D_ND, self).__init__(**kwargs)
                self.filters = filters
                self.act = act
            def build(self, input_shape):
                self.conv = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=3, activation=self.act)
            def call(self, inputs):
                shp = inputs.shape
                x = tf.reshape(inputs, shape=[-1, shp[-2], shp[-1]])
                x = self.conv(x)
                x = tf.reshape(x, shape=[-1] + [s for s in shp[1:-2]] + [x.shape[-2], self.filters])
                return x

        x_input = tf.keras.layers.Input(shape=[self.mb_of_same_realizations, self.len_timeseries, self.num_input_channels], name='x_observation')
        x = x_input
        self.num_conv_filters = [[16, 16], [32, 32]]
        for i in range(len(self.num_conv_filters)):
            if i != 0:
                x = tf.keras.layers.Lambda(maxpool1D_nd_fn, name='maxpool%d'%(i+1))(x)
            for j in range(len(self.num_conv_filters[i])):
                x = Conv1D_ND(filters=self.num_conv_filters[i][j], act='relu', name='conv%d_%d'%((i+1), (j+1)))(x)
        x = Conv1D_ND(filters=self.ndims_latent, act=None, name='final_conv')(x)
        latent_space = tf.keras.layers.Lambda(lambda x:tf.math.reduce_mean(x, axis=-2), name='global_avg_pool')(x) #shape: [num_different_mb, mb_of_same_realizations, ndims_latent]
        return tf.keras.Model(inputs=x_input, outputs=latent_space)

    def aggregator_fn(self, num_free_parameters):
        '''latent_mappings size: [bs, #ndims_latent]
        output: [#model_parameters]'''
        if self.ndims_latent <= num_free_parameters:
            raise AssertionError('#latent dimensions is <= #free_parameters. This makes no sense! You are doing something wrong.')
        # tile latent_mappings to timeseries length of the noise vectors.
        latent_mappings = tf.keras.layers.Input(shape=[self.mb_of_same_realizations, self.ndims_latent], name='latent_representations')
        # Define number of nodes in intermediate hidden layers. Uses a heuristic to pick based on number of free parameters. 
        if num_free_parameters < 2:
            num_ihn = [3, 10, 3]
        elif num_free_parameters < 10:
            num_ihn = [10, 100, 10]
        else: 
            raise AssertionError('Number of free parameters %d is too large! You are probably doing something wrong.' % num_free_parameters)
        slice_op = tf.keras.layers.Lambda(lambda x:x[..., self.ndims_latent-num_free_parameters:], name='slice_free_parameters')
        x = slice_op(latent_mappings)
        for i in range(len(num_ihn)):
            x = tf.keras.layers.Dense(units=num_ihn[i], activation=tf.keras.layers.LeakyReLU(alpha=0.3), name='fc_%d'%(i+1))(x)
        weighting_vector = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid, name='fc_%d'%(len(num_ihn)+1))(x) # size: [num_different_mb, mb_of_same_realizations, 1]
        weighting_vector = tf.clip_by_value(weighting_vector, clip_value_min=1e-8, clip_value_max=tf.math.reduce_max(weighting_vector))
        weighting_vector = weighting_vector / tf.math.reduce_sum(weighting_vector, axis=1, keepdims=True) # scale weighting vector such that it adds up to 1. Small epsilon for stability. 
        scale_op = tf.keras.layers.Lambda(lambda x: x[0][..., :self.ndims_latent-num_free_parameters] * x[1], name='scale_model_parameters')
        scaled_model_params = scale_op((latent_mappings, weighting_vector)) # size: [num_different_mb, mb_of_same_realizations, num_model_parameters]

        weighted_average = tf.math.reduce_sum(scaled_model_params, axis=1, keepdims=False, name='pred_model_parameters') # size: [num_different_mb, num_model_parameters]
        return tf.keras.Model(inputs=latent_mappings, outputs=weighted_average)

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
        self.logdir = '/tmp/model1_INCA'
        self.ndims_latent = 3 # #summary stats
        self.num_noise_channels = 1 # #different noise vectors
        self.num_model_parameters = 2 # NLAR1 model has c and sigma as model parameters.
        self.len_timeseries = 200
        self.batch_size = 5 # number of samples drawn from the same model parameter realizations
        self.num_different_mb = 60
        self.max_training_steps = int(3*1e8) # maximum number of training steps unless optimization gets killed.
        self.freq_log = 100 # frequency to update logged values in tensorboard.
        self.linear_warmup_steps = 1 # int(5*1e3) # #steps with linear warmup from very low lr to init_lr

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
    gen_train = src.generators.DataGenerator_NLAR1_Simplified_BatchSampler(len_timeseries=args.len_timeseries, batch_size=args.batch_size, x_0=x_0, c_lims=c_lims, sigma_lims=sigma_lims, fn=fn)
    dataset_train = tf.data.Dataset.from_generator(lambda: gen_train, output_types=(tf.float32, tf.float32, tf.float32))
    dataset_train = dataset_train.repeat(count=1)
    dataset_train = dataset_train.batch(args.num_different_mb, drop_remainder=True) # shape: [num_different_mb, batch_size, ...]
    dataset_train = dataset_train.prefetch(buffer_size=10) # #number of minibatches to pre-fetch.

    ##################################################################################################
    # Define the architecture
    model_obj = Architecture(mb_of_same_realizations=args.batch_size, ndims_latent=args.ndims_latent, ndims_free_latent_parameters=args.ndims_latent-args.num_model_parameters, len_timeseries=args.len_timeseries, num_noise_channels=args.num_noise_channels, ndims_output=args.num_model_parameters)
    model_obj.encoder.summary()
    model_obj.aggregator.summary()

    ##################################################################################################
    # Define objective functions
    class ChiSquareStatistic(tf.keras.losses.Loss):
        @tf.function
        def call(self, y_true, y_pred):
            '''Implements \sum( (y_true - y_pred) ** 2 / y_true **2.
            Expects rank1 array (num_different_mb) y_pred for theta, and rank 2 array (num_different_mb, same_mb) y_pred for latent_z'''
            if tf.not_equal(tf.rank(y_true), tf.rank(y_pred)):
                y_true = tf.expand_dims(y_true, axis=1)
            sd = tf.math.squared_difference(y_true, y_pred)
            if tf.rank(sd) > 1:
                return tf.math.reduce_sum(sd / tf.math.maximum(tf.math.pow(y_true, 2), 1e-6), axis= -1)
            else:
                return sd / tf.math.maximum(tf.math.pow(y_true, 2), 1e-6)

    @tf.function
    def loss_aggregator_fn(params, theta_pred, return_each_dim=False):
        # params.shape: [args.num_different_mb, args.num_model_parameters]
        # theta_pred.shape: [args.num_different_mb, args.num_model_parameters]
        num_params = params._shape_as_list()[-1] 
        if return_each_dim:
            d = {}
            for i in range(params.shape[-1]):
                d['ChiSquare_theta_ch_%d'%(i+1)] = ChiSquareStatistic(name='ChiSquare_theta_%d'%(i+1))(y_true=params[:,i], y_pred=theta_pred[:,i]) / args.num_model_parameters 
            return d
        else:
            loss = 0.
            for i in range(num_params):
                loss += ChiSquareStatistic(name='ChiSquare_theta_%d'%(i+1))(params[:,i], theta_pred[:,i]) / args.num_model_parameters
            return loss
    @tf.function
    def loss_regress_params_fn(params, params_pred, return_each_dim=False):
        ''' Notice params_pred can have more dimensions than params (free dimensions). This function is to be used for latent dims (z_latent)
        If not using this loss, return 0. instead.'''
        # params.shape: [args.num_different_mb, args.num_model_parameters]
        # params_pred.shape: [args.num_different_mb, args.batch_size, args.ndims_latent]
        num_params = params._shape_as_list()[-1] 
        if return_each_dim:
            d = {}
            for i in range(num_params):
                # d['MSE_z_%d'%(i+1)] = tf.keras.losses.MeanSquaredError(name='loss_param_%d'%(i+1))(params[...,i], params_pred[...,i]) / args.num_model_parameters 
                d['ChiSquare_z_%d'%(i+1)] = ChiSquareStatistic(name='ChiSquare_z_%d'%(i+1))(params[...,i], params_pred[...,i]) / args.num_model_parameters
            return d
        else:
            loss = 0.
            for i in range(num_params):
                # loss += tf.keras.losses.MeanSquaredError(name='loss_param_%d'%(i+1))(params[...,i], params_pred[...,i]) / args.num_model_parameters 
                loss += ChiSquareStatistic(name='ChiSquare_z_%d'%(i+1))(params[...,i], params_pred[...,i]) / args.num_model_parameters
            return loss
    # Define additional metrics for logging at every

    ##################################################################################################
    # Define optimizer 
    # lr_schedule = src.utils_tf.LearningRateScheduleExponentialDecayWithLinearWarmup(steps_warmup=args.linear_warmup_steps, initial_learning_rate=1.e-3, decay_steps=int(6*1e3), decay_rate=0.92, staircase=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.e-3, decay_steps=int(6*1e3), decay_rate=0.92, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    global_gradient_clipnorm = None
    ##################################################################################################
    # Define ckpt managers to save model weights throughout optimization
    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=model_obj.encoder, aggregator=model_obj.aggregator)
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
            z_latent = model.encoder(x, training=True) # shape: [num_different_mb, batch_size, num_latent_dims]
            theta_reconst = model.aggregator(z_latent, training=True) # shape: [num_different_mb, num_model_params]
            lambda_theta = args.batch_size
            lambda_z = 1.
            dict_regress_theta_mse = loss_aggregator_fn(params, theta_reconst, return_each_dim=True)
            loss_regress_theta = tf.math.reduce_sum(list(dict_regress_theta_mse.values()))
            dict_regress_z_mse = loss_regress_params_fn(params, z_latent, return_each_dim=True)
            loss_regress_z = tf.math.reduce_sum(list(dict_regress_z_mse.values()))
            loss = lambda_theta*loss_regress_theta + lambda_z*loss_regress_z
            trainable_variables = model.encoder.trainable_variables + model.aggregator.trainable_variables
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
        return (loss_regress_theta, loss_regress_z), (theta_reconst, z_latent), (dict_regress_theta_mse, dict_regress_z_mse)
    
    ##################################################################################################
    # Define a few metrics to export to tensorboard
    avg_loss_total = tf.keras.metrics.Mean(name='loss_total', dtype=tf.float32) # variable will keep track of average of loss value since last freq_log
    avg_loss_reg_theta = tf.keras.metrics.Mean(name='loss_regress_theta', dtype=tf.float32)
    avg_loss_reg_z = tf.keras.metrics.Mean(name='loss_regress_z', dtype=tf.float32)
    dict_avg_loss_reg_theta_items = {}
    dict_avg_loss_reg_z_items = {}
    dict_avg_rmse_reg_theta_items = {}
    dict_avg_rmse_reg_z_items = {}

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
        loss_tuple, x_theta_z, dict_mse = train_step(model=model_obj, x=x, params=params, noise=noise, optimizer=optimizer)
        theta_reconst, z_latent = x_theta_z
        loss_regress_theta, loss_regress_z = loss_tuple
        loss_total = loss_regress_theta + loss_regress_z
        dict_regress_theta_mse, dict_regress_z_mse = dict_mse
        
        # Update loggers for tensorboard
        avg_loss_reg_theta.update_state(loss_regress_theta)
        avg_loss_reg_z.update_state(loss_regress_z)
        avg_loss_total.update_state(loss_total) # aggregate values since last flush
        avg_loss_long_term.update_state(loss_total)
        for k in dict_regress_theta_mse:
            if k not in dict_avg_loss_reg_theta_items:
                dict_avg_loss_reg_theta_items[k] = tf.keras.metrics.Mean(name=k, dtype=tf.float32)
            dict_avg_loss_reg_theta_items[k].update_state(dict_regress_theta_mse[k])
        for k in dict_regress_z_mse:
            if k not in dict_avg_loss_reg_z_items:
                dict_avg_loss_reg_z_items[k] = tf.keras.metrics.Mean(name=k, dtype=tf.float32)
            dict_avg_loss_reg_z_items[k].update_state(dict_regress_z_mse[k])
                
        for i_ in range(params.shape[-1]):
            k = 'RMSE_theta_ch_%d'%(i_+1)
            if k not in dict_avg_rmse_reg_theta_items:
                dict_avg_rmse_reg_theta_items[k] = tf.keras.metrics.RootMeanSquaredError(name='rmse_theta', dtype=tf.float32)
            dict_avg_rmse_reg_theta_items[k].update_state(y_true=params[:,i_], y_pred=theta_reconst[:,i_]) # params.shape: [num_different_mb, num_model_params], theta_reconst.shape: [num_different_mb, num_model_params]
        for i_ in range(params.shape[-1]):
            k = 'RMSE_z_ch_%d'%(i_+1)
            if k not in dict_avg_rmse_reg_z_items:
                dict_avg_rmse_reg_z_items[k] = tf.keras.metrics.RootMeanSquaredError(name='rmse_z', dtype=tf.float32)
            dict_avg_rmse_reg_z_items[k].update_state(y_true=params[...,i_:i_+1], y_pred=z_latent[...,i_]) # params.shape: [num_different_mb, num_model_params], z_latent.shape: [num_different_mb, args.batch_size, ndims_latent]
        # Export status to tensorboard
        if tf.equal(optimizer.iterations % args.freq_log, 0):
            d_scalars = {'loss_total': avg_loss_total.result(), 'loss_regress_theta': avg_loss_reg_theta.result(), 'loss_regress_z': avg_loss_reg_z.result()}
            for k in dict_avg_loss_reg_theta_items:
                d_scalars[k] = dict_avg_loss_reg_theta_items[k].result()
                dict_avg_loss_reg_theta_items[k].reset_states()
            for k in dict_avg_loss_reg_z_items:
                d_scalars[k] = dict_avg_loss_reg_z_items[k].result()
                dict_avg_loss_reg_z_items[k].reset_states()
            for k in dict_avg_rmse_reg_theta_items:
                d_scalars[k] = dict_avg_rmse_reg_theta_items[k].result()
                dict_avg_rmse_reg_theta_items[k].reset_states()
            for k in dict_avg_rmse_reg_z_items:
                d_scalars[k] = dict_avg_rmse_reg_z_items[k].result()
                dict_avg_rmse_reg_z_items[k].reset_states()
            d_scalars['lr_schedule'] = lr_schedule(step=num_step)
            export_summary_scalars(dict_name_and_val=d_scalars, step=optimizer.iterations, writer=summary_writer)
            # Print current loss status to terminal
            logging.info('Step %d: avg loss: %.3f, parameter (theta) regression loss: %.3f, latent_z regression loss: %.3f.' % \
                    (num_step, d_scalars['loss_total'], d_scalars['loss_regress_theta'], d_scalars['loss_regress_z']))
            avg_loss_total.reset_states() #reset kept history of loss
            avg_loss_reg_theta.reset_states()
            # Export trainable variables to tboard histogram
            # TODO: consider speeding this up.
            l_enc = list(zip(*[['enc_'+v.name, v.value()] for v in model_obj.encoder.trainable_variables]))
            l_agg = list(zip(*[['agg_'+v.name, v.value()] for v in model_obj.aggregator.trainable_variables]))
            d_histograms = {**dict(zip(l_enc[0], l_enc[1])), **dict(zip(l_agg[0], l_agg[1]))}
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
    def __init__(self, generator=None, iterator=None,  mb_of_same_realizations=1, **kwargs):
        '''Constructor builds NN model and loads its weights. In addition, a generator with true sun parameters is initialized.'''
        self.args = ExpSetup() # get experiment parameters from the ExpSetup class in this file. Make sure logdir is accurate.
        self.mb_of_same_realizations = mb_of_same_realizations
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

        l_batches = []
        for i in range(num_samples):
            l_batches.append(next(iterator)) # sample contains a tuple of form (x, params, noise)
        observations, model_params, noise = [tf.stack(it, axis=0) for it in list(zip(*l_batches))]
        summary_space = self.model_obj.encoder(observations, training=False).numpy()
        ndarray_noise_timeseries = noise.numpy()
        model_params = model_params.numpy()
        observations = np.squeeze(observations.numpy())
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
        o_latent = self.model_obj.encoder(samples, training=False)
        return o_latent

    def decode(self, summary_stats):
        '''Function reconstruct timeseries for a given tuple of (latent_representations, noise vectors).'''
        o_theta = self.model_obj.aggregator(summary_stats, training=False).numpy()
        return o_theta

    def check_hyper_params(self):
        hp_manager = Manage_Hyper_Parameters(logdir=self.args.logdir)
        if hp_manager.args is None:
            raise AssertionError('Hyper-parameter configuration file %s is not found. Quitting.' % (hp_manager.param_config_fn))
        else:
            for attr in dir(self.args):
                if not attr.startswith('__') and attr not in ['logdir', 'batch_size']:  # don't get methods, ignore logdir and batch_size, as the optimization could have been done elsewhere.
                    if not hasattr(hp_manager.args, attr):
                        print('Saved ExpSetup file is missing attribute %s. Will ignore.' % str(attr))
                    else:
                        if getattr(self.args, attr) != getattr(hp_manager.args, attr):
                            raise AssertionError('Mismatch of hyper-parameter attributes in the logdir %s and ExpSetup file in this script for attribute: %s' % (self.args.logdir, attr))

    def build_model(self):
        self.model_obj = Architecture(mb_of_same_realizations=self.mb_of_same_realizations, ndims_latent=self.args.ndims_latent, len_timeseries=self.args.len_timeseries, num_noise_channels=self.args.num_noise_channels, ndims_output=self.args.num_model_parameters, ndims_free_latent_parameters=self.args.ndims_latent-self.args.num_model_parameters)

    def load_model(self, basename='model_best_ckpt'):
        ckpt = tf.train.Checkpoint(encoder=self.model_obj.encoder, aggregator=self.model_obj.aggregator)
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
        batch_size_requested = kwargs.pop('batch_size', None)
        if batch_size_requested is not None:
            if batch_size_requested != self.mb_of_same_realizations:
                print('A separate "batch_size" parameter sent to the data generator. This is expected to match mb_of_same_realizations (N), but it does not, hence it is ignored.')
        generator = src.generators.DataGenerator_NLAR1_Simplified_BatchSampler(len_timeseries=self.args.len_timeseries, batch_size=self.mb_of_same_realizations, prng=prng, x_0=x_0, c_lims=c_lims, sigma_lims=sigma_lims, fn=func, **kwargs) ## Although we can request mb_of_same_realizations custom, it's safer to fix it to initial setup, which will hopefully match the number used in training of the model.
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
