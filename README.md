This is a very rough prototype of a pet project. The idea is to train a
recurrent neural network (an [LSTM](https://en.wikipedia.org/wiki/LSTM),
specifically) to infer past population sizes from allele frequencies and their
spacings across the genome, sampled from present-day individuals. In its
objective and the data it uses it is thus very similar to Jonathan Terhorst's
[SMC++](https://www.ncbi.nlm.nih.gov/pubmed/28024154) method (code
[here](https://github.com/popgenmethods/smcpp)).

The approach I am considering is to simulate data from randomly generated
demographies (defined in this problem as a piecewise-constant trajectory of
population sizes changing at different fixed times in the past, a la
[PSMC](https://www.nature.com/articles/nature10231)), with the
relative population sizes changing according to an exponentiated Gaussian
process. For each simulated demography, we simulate allele frequencies and
their spacings across a chromosome, producing features corresponding to the
vector of "labels" comprised of the population sizes at each time in the past.
The NN outputs gamma distribution parameters for each timepoint in the past, so
that the loss function is the sum of the negative log-likelihoods of the true
relative population sizes given the gamma distributions output by the NN.

Simulations are perfomed using
[`msprime`](https://github.com/tskit-dev/msprime) and the RNN is implemented
using Tensorflow's implementation of the [Keras
API](https://www.tensorflow.org/guide/keras) (with a Distribution layer from
[tensorflow_probability](https://www.tensorflow.org/probability)).
