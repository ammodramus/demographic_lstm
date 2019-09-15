# coding: utf-8

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as npr
import re
from numba import jit
import tensorflow as tf
import multiprocessing as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Lambda, Masking
from tensorflow.keras.layers import LSTM
import argparse
import os

import msprime

parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--load', help = 'load the model from a file')
args = parser.parse_args()


# # generating data for demographic inference with RNNs
# 
# We want to generate genotype (and allele frequency, and spacing) data for a
# demographic-inference RNN. The first thing we'll try is generating lots of
# chromosome-scale data from populations with different size histories. 
# 
# 
# 
# We'll assume that population size is piecewise-constant, according to the
# original PSMC time-interval scheme. Underlying the changes in population size
# will be a [geometric Brownian motion
# (GBM)](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) process,
# where the population size for a given time interval is equal to the value of
# the GBM process at the midpoint of that time interval. 
# 
# The PSMC time interval scheme is
# 
# $$t_j = 0.1(e^{\frac{j}{n} \log(1+10T_{max})}-1),$$
# 
# where time is measured in units of effective population size, $N_e$.
# 
#  - 

# Set the boundary of the last time interval.
# Farther back in time than this, the population size is constant.
tmax = 10
num_intervals = 20
# See p. 21 of the Li and Durbin (2011, the original PSMC paper) supplement
# The one change here is that we go one past t_max in order to get a point
# to sample the final population size.
time_boundaries = 0.1*(np.exp(np.arange(num_intervals+2)/float(num_intervals)*np.log(1+10*tmax))-1)

# a function to produce one-dimensional Brownian motion
@jit
def brownian_motion(x0, ts, sigma2):
    '''
    x0      initial value
    ts      time boundaries, must begin with the first non-zero value
    sigma2  variance parameter, in the same units as t
    '''
    x = np.zeros(ts.shape[0])
    sigma = np.sqrt(sigma2)
    x[0] = npr.normal(x0, sigma*(ts[0]))
    for i in range(1,ts.shape[0]):
        x[i] = npr.normal(x[i-1], sigma*(ts[i]-ts[i-1]))
    return x

time_midpoints = time_boundaries[:-1] + (time_boundaries[1:] - time_boundaries[:-1])/2.0
#time_midpoints = np.concatenate(([0], time_midpoints))
num_pop_sizes = time_midpoints.shape[0]

data_queue = mp.Queue(256)

num_data_threads = 12

sample_size = 1000
#mutation_rate = 1000
#recombination_rate = 1000
#to_keep = 500

#chromosome_length = 1e8  # 100 Megabases
#chromosome_length = 1e7  # 10 Megabases
chromosome_length = 1e6  # 1 Megabase
position_mean = chromosome_length/2.0
position_std = np.sqrt(1.0/12)*chromosome_length

lower_mutation_rate = 1e-9*1e4
upper_mutation_rate = 1e-8*1e4
lower_recombination_rate = 1e-9*1e4
upper_recombination_rate = 1e-8*1e4

mutations_per_tree = 1

# calibration_afs = []
# for i in range(100):
#     pop_sizes = np.exp(brownian_motion(0.0, time_midpoints, 1))
#     demo_events = [
#         msprime.PopulationParametersChange(t, size)
#         for t, size in zip(time_midpoints, pop_sizes)]
#     mutation_rate = 10**npr.uniform(np.log10(lower_mutation_rate),
#                                 np.log10(upper_mutation_rate))
#     trees = msprime.simulate(sample_size=sample_size,
#                              Ne=1,
#                              length=chromosome_length,
#                              num_replicates=10,
#                              mutation_rate=mutation_rate)
#     for j, tree in enumerate(trees):
#         calibration_afs.extend(list(tree.genotype_matrix().sum(1) /
#                                     sample_size))
#         print(i, j)
# print(np.mean(calibration_afs), np.std(calibration_afs))
# import pdb; pdb.set_trace()

af_mean = 0.1334
af_std = 0.2227

def produce_data(queue, thread_id, sample_size, mutation_rate, recombination_rate, time_midpoints, to_keep):
    npr.seed(thread_id*os.getpid())
    while True:
        pop_sizes = np.exp(brownian_motion(0.0, time_midpoints, 1))
        demo_events = [msprime.PopulationParametersChange(t, size) for t, size in zip(time_midpoints, pop_sizes)]
        trees = msprime.simulate(sample_size=sample_size,
                                 Ne=1,
                                 length=1,
                                 recombination_rate=1000,
                                 mutation_rate=1000)
        gens = trees.genotype_matrix()
        afs = gens.sum(1)/sample_size
        positions = [var.site.position for var in trees.variants()]
        tdat = np.transpose(np.vstack((afs[:to_keep], positions[:to_keep])))
        queue.put((tdat, pop_sizes))

def produce_data_2(
        queue,
        thread_id,
        sample_size,
        time_midpoints,
        lower_recombination_rate,
        upper_recombination_rate,
        lower_mutation_rate,
        upper_mutation_rate,
        mutations_per_tree):

    npr.seed(thread_id)
    while True:
        pop_sizes = np.exp(brownian_motion(0.0, time_midpoints, 1))
        demo_events = [
            msprime.PopulationParametersChange(t, size)
            for t, size in zip(time_midpoints, pop_sizes)]
        recombination_rate = 10**npr.uniform(np.log10(lower_recombination_rate),
                                         np.log10(upper_recombination_rate))
        trees = msprime.simulate(sample_size=sample_size,
                                 Ne=1,
                                 length=chromosome_length,
                                 recombination_rate=recombination_rate)
        for i in range(mutations_per_tree):
            mutation_rate = 10**npr.uniform(np.log10(lower_mutation_rate),
                                        np.log10(upper_mutation_rate))
            mut_trees = msprime.mutate(trees, rate=mutation_rate)
            gens = mut_trees.genotype_matrix()
            afs = gens.sum(1)/sample_size
            norm_afs = (afs-af_mean)/af_std
            positions = np.array([var.site.position for var in
                                  mut_trees.variants()])
            norm_positions = (positions-position_mean)/position_std
            tdat = np.transpose(np.vstack((norm_afs, norm_positions)))
            queue.put((tdat, pop_sizes))

data_processes = [
    mp.Process(target=produce_data_2,
               args=(
                   data_queue,
                   tid,
                   sample_size,
                   time_midpoints,
                   lower_recombination_rate,
                   upper_recombination_rate,
                   lower_mutation_rate,
                   upper_mutation_rate,
                   mutations_per_tree))
                   for tid in range(num_data_threads)]



for p in data_processes:
    p.start()

def data_generator():
    while True:
        tdat, pop_sizes = data_queue.get()
        yield tdat, pop_sizes

batch_size = 32

dataset = tf.data.Dataset.from_generator(
    data_generator, output_types =
    (tf.float32, tf.float32),
    output_shapes = (tf.TensorShape([None, 2]),
                     tf.TensorShape([num_pop_sizes])))
dataset = dataset.padded_batch(batch_size, ((tf.compat.v1.Dimension(None),
                                    tf.compat.v1.Dimension(2)),
                                    tf.compat.v1.Dimension(num_pop_sizes)))

diter = dataset.make_one_shot_iterator()
dnext = diter.get_next()
with tf.Session() as sess:
    afs, trajs = sess.run(dnext)

if args.load is not None:
    model = tf.keras.models.load_model(args.load_model)
else:
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(LSTM(128, input_shape=(None,2)))
    model.add(Dense(2*num_pop_sizes))
    model.add(Lambda(lambda x: x*x))


def gamma_mse_loss(y_true, y_pred):
    '''
    E[(X-a)^2] = E[X^2] - 2aE[X] + a^2

    E[X^2] = Var[X] + E[X]^2

    Note: To get around the annoying fact that y_true and y_pred have to be the
    same dimension, we pad y_true with an extra complement of zeros. The first
    num_pop_sizes are the true population sizes, and the second num_pop_sizes
    are just zero padding.

    (I thought we'd have to do this, but it turns out the requirement for
    y_true and y_pred having the same shape doesn't seem to hold.)

    Initially was parameterizing with gamma distribution parameters, but
    hitting a "mean" with that requires coordinating two outputs.
    Reparameterizing with mean and var/mean, which should be easier to fit.
    '''
    #shapes = y_pred[:,:num_pop_sizes]
    #scales = y_pred[:,num_pop_sizes:]
    #means = shapes*scales
    #variances = means*scales
    y_true = tf.keras.backend.print_tensor(y_true, 'y_true: ')
    #y_pred = tf.keras.backend.print_tensor(y_pred, 'y_pred: ')

    means = y_pred[:,:num_pop_sizes]
    means = tf.keras.backend.print_tensor(means, "means: ")
    var_over_means = y_pred[:,num_pop_sizes:]
    variances = var_over_means * means
    variances = tf.keras.backend.print_tensor(variances, "variances: ")
    second_moments = variances + means*means
    #second_moments = tf.keras.backend.print_tensor(second_moments, "second_moments: ")


    mses = (second_moments - 2*y_true*means + y_true*y_true)
    # Note, to try: try normalizing the MSE by the variance or SD.
    ret = tf.keras.backend.print_tensor(tf.reduce_sum(mses, axis=1),
                                        'loss values: ')
    return ret


def norm_mse_loss(y_true, y_pred):
    '''
    E[(X-a)^2] = E[X^2] - 2aE[X] + a^2

    E[X^2] = Var[X] + E[X]^2

    Note: To get around the annoying fact that y_true and y_pred have to be the
    same dimension, we pad y_true with an extra complement of zeros. The first
    num_pop_sizes are the true population sizes, and the second num_pop_sizes
    are just zero padding.

    (I thought we'd have to do this, but it turns out the requirement for
    y_true and y_pred having the same shape doesn't seem to hold.)

    Initially was parameterizing with gamma distribution parameters, but
    hitting a "mean" with that requires coordinating two outputs.
    Reparameterizing with mean and var/mean, which should be easier to fit.
    '''
    #y_true = tf.keras.backend.print_tensor(y_true, 'y_true: ')

    means = y_pred[:,:num_pop_sizes]
    #means = tf.keras.backend.print_tensor(means, "means: ")

    mses = (y_true - means)**2 / y_true
    #mses = tf.keras.backend.print_tensor(mses, 'mses: ')
    ret = tf.reduce_sum(mses, axis=1)
    #ret = tf.keras.backend.print_tensor(ret, 'loss values: ')
    return ret


#optimizer = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.0001)
optimizer = tf.keras.optimizers.Adam()
#model.compile(loss=gamma_mse_loss, optimizer=optimizer)
model.compile(loss=norm_mse_loss, optimizer=optimizer)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'test.model',load_weights_on_restart=True)
#tensorboard_callback = tf.keras.callbacks.Tensorboard('log/'
num_epochs = 100
batches_per_epoch = 100
model.fit(dataset, epochs=num_epochs, steps_per_epoch=batches_per_epoch,
          callbacks=[checkpoint_callback])

#model.predict(dataset.batch(16))
diter = dataset.make_one_shot_iterator()
dnext = diter.get_next()
with tf.Session() as sess:
    afs, trajs = sess.run(dnext)
prediction = model.predict(np.array([afs[0]]))
true_traj = trajs[0]
print(prediction)
print(true_traj)

for p in data_processes:
    p.terminate()
