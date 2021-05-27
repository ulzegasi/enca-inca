#Author: Firat Ozdemir, October 2019, firat.ozdemir@datascience.ch
import tensorflow as tf
from tensorflow.python.ops import math_ops

class LearningRateScheduleExponentialDecayWithLinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    @tf.function
    def __init__(self, steps_warmup, initial_learning_rate, decay_steps, decay_rate, staircase):
        super(LearningRateScheduleExponentialDecayWithLinearWarmup, self).__init__()
        self.steps_warmup = steps_warmup
        self.initial_learning_rate = initial_learning_rate
        self.warmup_start_lr = 1e-9
        self.slope = (self.initial_learning_rate - self.warmup_start_lr) / self.steps_warmup
        self.lr_expdecay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
    @tf.function
    def linear_warmup(self, step):
        return self.slope * (math_ops.cast(step, tf.float32)+1.) + self.warmup_start_lr
    @tf.function
    def __call__(self, step):
        if step < self.steps_warmup:
            return self.linear_warmup(step)
        else:
            return self.lr_expdecay(step-self.steps_warmup)
