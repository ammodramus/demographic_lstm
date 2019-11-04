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
import tensorflow_probability as tfp
import argparse
import os

import msprime

parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--load', help = 'load the model from a file')
parser.add_argument('--evaluate', action='store_true')
args = parser.parse_args()

tf.enable_eager_execution()

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
num_pop_sizes = time_midpoints.shape[0]

data_queue = mp.Queue(256)

num_data_threads = 12

sample_size = 1000

chromosome_length = 1e6  # 1 Megabase
position_mean = chromosome_length/2.0
position_std = np.sqrt(1.0/12)*chromosome_length

lower_mutation_rate = 1e-9*1e4
upper_mutation_rate = 1e-8*1e4
lower_recombination_rate = 1e-9*1e4
upper_recombination_rate = 1e-8*1e4

mutations_per_tree = 1

af_mean = 0.1334
af_std = 0.2227

def produce_data(
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
    mp.Process(target=produce_data,
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

if args.load is not None:
    model = tf.keras.models.load_model(args.load)
else:
    model = Sequential()
    model.add(Masking(mask_value=0))
    model.add(LSTM(128, input_shape=(None,2)))
    model.add(Dense(2*num_pop_sizes))
    model.add(Lambda(lambda x: x*x))
    model.add(tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Gamma(
            concentration=t[:,:num_pop_sizes]**2/t[:,num_pop_sizes:]+1e-5,
            rate=t[:,:num_pop_sizes]/t[:,num_pop_sizes:]+1e-5)))
    # Note on concentrations and rates [the NN outputs means (m) and
    # variances(v)]:
    # m = c/r
    # v = c/r^2
    # r = m/v
    # c = m*r = m*m/v



negloglik = lambda y, p_y: -tf.math.reduce_sum(p_y.log_prob(y))

optimizer = tf.keras.optimizers.Adam()
model.compile(loss=negloglik, optimizer=optimizer)

if args.evaluate:
    while True:
        tdat, pop_sizes = data_generator().next()
        tdat = tdat[np.newaxis,:,:].astype(np.float32)
        mean = model(tdat).mean()
        std = np.sqrt(model(tdat).variance())
        out = pd.DataFrame({'true': pop_sizes, 'predicted': mean[0],
                            'std': std[0]})
        print(out)
        import pdb; pdb.set_trace()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'tfp_gamma_fixed_loss.model',load_weights_on_restart=True)
num_epochs = 10000
batches_per_epoch = 100
model.fit(dataset, epochs=num_epochs, steps_per_epoch=batches_per_epoch,
          callbacks=[checkpoint_callback])
