################################################################################
#
# Copyright (c) 2016 eBay Software Foundation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#################################################################################
#
# @Author: Mingkuan Liu
# @Email:  mingkliu@ebay.com
# @Date:   2016-07-24
#
##################################################################################

"""

SSE (Sequence Semantic Embedding) is an encoder framework toolkit for NLP related tasks and it's implemented in
TensorFlow by leveraging TF's convenient DNN/CNN/RNN/LSTM building blocks.

SSE model translates a sequence of symbols into a vector of numeric numbers, so that different sequences
with similar semantic meanings will have closer numeric vector distances. This numeric number vector is
called the SSE for the original sequence of symbols.

SSE can be applied to some large scale NLP related machine learning tasks. For example, it can be applied to
large scale classification task: mapping a listing title to one of the 20,000+ leaf categories in eBay website.
Or it can be applied to information retrieval task: mapping a search query to some most relevant documents in the inventory.

Depending on each specific task, similar semantic meanings can have different definitions. For example,
in a classification task, similar semantic meanings means that for each correct pair of (listing-title, category),
the SSE of title is close to the SSE of corresponding category.  While in an information retrieval task,
similar semantic meaning means for each relevant pair of (query, document), the SSE of query is close to the SSE of
relevant document.


SSE encoder framework supports three different types of network configuration modes: source-encoder-only, dual-encoder
and shared-encoder.

* In source-encoder-only mode, SSE will only train a single encoder model(RNN/LSTM/CNN) for source sequence.
For target sequence, SSE will just learn its sequence embedding directly without applying any encoder models.
This mode is suitable for closed target space tasks such as classification task, since in such tasks the target
sequence space is limited and closed thus it does not require to generate new embeddings for any future unknown
target sequences outside of training stage.

* In dual-encoder mode, SSE will train two different encoder models(RNN/LSTM/CNN) for both source sequence and
target sequence. This mode is suitable for open target space tasks such as information retrieval, since the target
sequence space in those tasks is open and dynamically changing, a specific target sequence encoder model is needed to
generate embeddings for new unobserved target sequence outside of training stage.

* In shared-encoder mode, SSE will train one single encoder model(RNN/LSTM/CNN) shared for both source sequence
and target sequence. This mode is suitable for open target space tasks such as question answering system or
information retrieval system, since the target sequence space in those tasks is open and dynamically changing,
a specific target sequence encoder model is needed to generate embeddings for new unobserved target sequence
outside of training stage. In shared-encoder mode, the source sequence encoder model is the same as target
sequence encoder mode. Thus this mode is better for tasks where the vocabulary between source sequence and
target sequence are similar and can be shared.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils


class SSEModel(object):

  # Number of target sequences returned by prediction operation

  def __init__(self, MAX_SEQ_LENGTH, max_gradient_norm, src_vocab_size, tgt_vocab_size, word_embed_size, seq_embed_size,
               src_cell_size, tgt_cell_size, num_layers,
               learning_rate, learning_rate_decay_factor, targetSpaceSize, network_mode='source-encoder-only', forward_only=False, TOP_N=20, name="SSEModel" ):


    """ Create the Sequence Semantic Embedding Model.

    :param src_vocab_size: source data vocab size
    :param tgt_vocab_size:  target data vocab size
    :param word_embed_size: dim for token level word embedding
    :param seq_embed_size:  dim for sequence level embedding
    :param src_cell_size:  source model RNN cell size
    :param tgt_cell_size:  target model RNN cell size
    :param num_layers:  number of layers for src/target model.
    :param learning_rate: learning rate to start with.
    :param learning_rate_decay_factor: decay learning rate by this much when needed
    :param network_mode: Config SSE network mode to be one of three modes: source-encoder-only, dual-encoder or shared-encoder.
    """

    # List of operations to be called after each training step, see
    # _add_post_train_ops
    self._post_train_ops = []
    self.name = name
    self.forward_only = forward_only
    self.network_mode = network_mode.strip()

    #default setup
    self.TOP_N = TOP_N

    #setup class parameters
    self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
    self.max_gradient_norm = max_gradient_norm
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.word_embed_size = word_embed_size
    self.seq_embed_size = seq_embed_size
    self.src_cell_size = src_cell_size
    self.tgt_cell_size = tgt_cell_size
    self.num_layers = num_layers
    self.learning_rate = tf.Variable(float(learning_rate), name='learning_rate', trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.targetSpaceSize = targetSpaceSize


    # setup basic model cell type to be LSTM or GRU or CNN
    # TODO: enhence for CNN basic unit later
    self.use_lstm = True

    # Setup Source internal RNN Cell in tensoflow graph
    self._create_embedders()
    self._def_loss()
    self._def_optimize()
    self._def_predict()

    self.saver = tf.train.Saver(tf.all_variables() , max_to_keep=20)

  @staticmethod
  def _last_relevant(output, length):
    b_size = tf.shape(output)[0]
    max_len = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    idx = tf.range(0,b_size) * max_len + (length -1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, idx)
    return relevant

  def _create_embedders(self):

    #placeholder for input data
    self._src_input_data = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='source_sequence')
    self._tgt_input_data = tf.placeholder(tf.int32, [None, self.MAX_SEQ_LENGTH], name='target_sequence')
    self._labels = tf.placeholder(tf.int64, [None], name='targetSpace_labels')
    self._src_lens = tf.placeholder(tf.int32, [None], name='source_seq_lenths')
    self._tgt_lens = tf.placeholder(tf.int32, [None], name='target_seq_lenths')

    #create word embedding vectors
    self.src_word_embedding = tf.get_variable('src_word_embedding', [self.src_vocab_size, self.word_embed_size],
                                         initializer=tf.random_uniform_initializer(-0.25,0.25))

    self.tgt_word_embedding = tf.get_variable('tgt_word_embedding', [self.tgt_vocab_size, self.word_embed_size],
                                         initializer=tf.random_uniform_initializer(-0.25, 0.25))

    #transform input tensors from tokenID to word embedding
    self.src_input_distributed = tf.nn.embedding_lookup( self.src_word_embedding, self._src_input_data, name='dist_source')
    self.tgt_input_distributed = tf.nn.embedding_lookup( self.tgt_word_embedding, self._tgt_input_data, name='dist_target')


    if self.network_mode == 'source-encoder-only':
      self._source_encoder_only_network()
    elif self.network_mode == 'dual-encoder':
      self._dual_encoder_network()
    elif self.network_mode == 'shared-encoder':
      self._shared_encoder_network()
    else:
      print('Error!! Unsupported network mode: %s. Please specify on: source-encoder-only, dual-encoder or shared-encoder.' % self.network_mode )
      exit(-1)



  def _source_encoder_only_network(self):
    # config SSE network to be source encoder only mode
    # Build source encoder
    with tf.variable_scope('source_only_encoder'):
      # TODO: need play with forgetGate and peeholes here
      if self.use_lstm:
        src_single_cell = tf.nn.rnn_cell.LSTMCell(self.src_cell_size, forget_bias=1.0, use_peepholes=False)
      else:
        src_single_cell = tf.nn.rnn_cell.GRUCell(self.src_cell_size)
      src_cell = src_single_cell
      if self.num_layers > 1:
        src_cell = tf.nn.rnn_cell.MultiRNNCell([src_single_cell] * self.num_layers)

      src_output, _ = tf.nn.dynamic_rnn(src_cell, self.src_input_distributed, sequence_length=self._src_lens,
                                        dtype=tf.float32)
      src_last_output = self._last_relevant(src_output, self._src_lens)
      self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                   initializer=tf.truncated_normal_initializer())
      # self.src_b = tf.get_variable('src_b', shape=[self.seq_embed_size])
      self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)  # + self.src_b

    # Build target encoder
    with tf.variable_scope('target_embedding'):
      # no need train target model, just train embedding directly
      self.tgt_seq_embedding = tf.get_variable('tgt_seq_embedding', shape=[self.targetSpaceSize, self.seq_embed_size],
                                                 initializer=tf.random_uniform_initializer(-0.25, 0.25))



  def _dual_encoder_network(self):
    # config SSE network to be dual encoder mode
    # Build source encoder
    with tf.variable_scope('source_encoder'):
      # TODO: need play with forgetGate and peeholes here
      if self.use_lstm:
        src_single_cell = tf.nn.rnn_cell.LSTMCell(self.src_cell_size, forget_bias=1.0, use_peepholes=False)
      else:
        src_single_cell = tf.nn.rnn_cell.GRUCell(self.src_cell_size)
      src_cell = src_single_cell
      if self.num_layers > 1:
        src_cell = tf.nn.rnn_cell.MultiRNNCell([src_single_cell] * self.num_layers)

      src_output, _ = tf.nn.dynamic_rnn(src_cell, self.src_input_distributed, sequence_length=self._src_lens,
                                        dtype=tf.float32)
      src_last_output = self._last_relevant(src_output, self._src_lens)
      self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                   initializer=tf.truncated_normal_initializer())
      # self.src_b = tf.get_variable('src_b', shape=[self.seq_embed_size])
      self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)  # + self.src_b

    # Build target encoder
    with tf.variable_scope('target_encoder'):
      # need train target model
      # TODO: need play with forgetGate and peeholes here
      tgt_single_cell = tf.nn.rnn_cell.GRUCell(self.tgt_cell_size)
      if self.use_lstm:
        tgt_single_cell = tf.nn.rnn_cell.LSTMCell(self.tgt_cell_size, forget_bias=1.0, use_peepholes=False)
      tgt_cell = tgt_single_cell
      if self.num_layers > 1:
        tgt_cell = tf.nn.rnn_cell.MultiRNNCell([tgt_single_cell] * self.num_layers)

      tgt_output, _ = tf.nn.dynamic_rnn(tgt_cell, self.tgt_input_distributed, sequence_length=self._tgt_lens,
                                        dtype=tf.float32)
      tgt_last_output = self._last_relevant(tgt_output, self._tgt_lens)
      self.tgt_M = tf.get_variable('tgt_M', shape=[self.tgt_cell_size, self.seq_embed_size],
                                   initializer=tf.truncated_normal_initializer())
      # self.tgt_b = tf.get_variable('tgt_b', shape=[self.seq_embed_size])
      self.tgt_seq_embedding = tf.matmul(tgt_last_output, self.tgt_M)  # + self.tgt_b



  def _shared_encoder_network(self):
    # config SSE network to be shared encoder mode
    # Build shared encoder
    with tf.variable_scope('shared_encoder'):
      # TODO: need play with forgetGate and peeholes here
      if self.use_lstm:
        src_single_cell = tf.nn.rnn_cell.LSTMCell(self.src_cell_size, forget_bias=1.0, use_peepholes=False)
      else:
        src_single_cell = tf.nn.rnn_cell.GRUCell(self.src_cell_size)

      src_cell = src_single_cell
      if self.num_layers > 1:
        src_cell = tf.nn.rnn_cell.MultiRNNCell([src_single_cell] * self.num_layers)

      #compute source sequence related tensors
      src_output, _ = tf.nn.dynamic_rnn(src_cell, self.src_input_distributed, sequence_length=self._src_lens,
                                        dtype=tf.float32)
      src_last_output = self._last_relevant(src_output, self._src_lens)
      self.src_M = tf.get_variable('src_M', shape=[self.src_cell_size, self.seq_embed_size],
                                   initializer=tf.truncated_normal_initializer())
      # self.src_b = tf.get_variable('src_b', shape=[self.seq_embed_size])
      self.src_seq_embedding = tf.matmul(src_last_output, self.src_M)  # + self.src_b

      #declare tgt_M tensor before reuse them
      self.tgt_M = tf.get_variable('tgt_M', shape=[self.src_cell_size, self.seq_embed_size],
                                   initializer=tf.truncated_normal_initializer())
      # self.tgt_b = tf.get_variable('tgt_b', shape=[self.seq_embed_size])

    with tf.variable_scope('shared_encoder', reuse=True):
      #compute target sequence related tensors by reusing shared_encoder model
      tgt_output, _ = tf.nn.dynamic_rnn(src_cell, self.tgt_input_distributed, sequence_length=self._tgt_lens,
                                        dtype=tf.float32)
      tgt_last_output = self._last_relevant(tgt_output, self._tgt_lens)

      self.tgt_seq_embedding = tf.matmul(tgt_last_output, self.tgt_M)  # + self.tgt_b



  def _def_loss(self):
    # compute src / tgt similarity
    with tf.variable_scope('similarity'):
      self.similarity = tf.matmul( self.src_seq_embedding, self.tgt_seq_embedding, transpose_b=True)
      # self.norm_similarity = tf.matmul( tf.nn.l2_normalize(self.src_seq_embedding, 1),
      #                                   tf.nn.l2_normalize( self.tgt_seq_embedding, 1), transpose_b=True)

    with tf.variable_scope('training_loss'):
      # # doing sampled softmax loss at here:
      # bias = tf.get_variable('loss_bias', shape=[self.targetSpaceSize], initializer=tf.truncated_normal_initializer() )
      # self.loss = tf.reduce_mean( tf.nn.sampled_softmax_loss( self.tgt_seq_embedding, bias, self.src_seq_embedding, \
      #         tf.expand_dims( self._labels, 1 ), self.neg_sample_size, self.targetSpaceSize) )

      #doing full target space softmax loss at here
      self.loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( self.similarity, self._labels) )

      #TODO: try sigmoid loss function later: tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)

  def set_top_n(self, top_n):
    self.TOP_N = top_n


  def set_forward_only(self, forward_only=True):
    self.forward_only = forward_only

  def _def_predict(self):
    # Prediction cannot return more candidates than there are categories
    #top_n = min( tf.shape(self._tgt_input_data)[0], SSEModel.TOP_N)
    with tf.name_scope("prediction"):
      self.predicted_tgts_score, self.predicted_labels = tf.nn.top_k(self.similarity, self.TOP_N, sorted=True)
      #normalize conf score
      self.predicted_tgts_score = tf.nn.l2_normalize(self.predicted_tgts_score, 1)
      #self.predicted_tgts_score, self.predicted_labels = tf.nn.top_k(self.norm_similarity, self.TOP_N, sorted=True)


  def _def_optimize(self):
    """
    Builds graph to minimize loss function.
    """

    #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    optimizer = tf.train.AdagradOptimizer(self.learning_rate)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_gradient_norm )
    self.train = optimizer.apply_gradients( zip(grads, tvars), global_step=self.global_step)  #tutorial version??

    # #TODO: try different optimizer to see if any improvements
    # self.train = optimizer.minimize(self.loss, global_step=self.global_step, gate_gradients=optimizer.GATE_NONE) #default version?

    self._add_post_train_ops()


  def _add_post_train_ops(self):
    """
    Replaces the self.train operation with an operation group, consisting of
    the training operation itself and the operations listed in
    self.post_train_ops.

    Called by _def_optimize().

    """
    with tf.control_dependencies([self.train]):
        self.train = tf.group(self.train, *self._post_train_ops)

  def save(self, session, path, global_step = None):
    """ Saves variables to given path """
    return self.saver.save(session, path, global_step)


  def load(self, session, path):
    """ Restores variables from given path """
    self.saver.restore(session, path)


  def add_summaries(self):
    """
    Adds summaries for the following variables to the graph and returns
    an operation to evaluate them.
     * loss (raw)
     * loss (moving average)

    """
    loss = tf.scalar_summary("loss (raw)", self.loss)
    return tf.merge_summary([loss])


  def get_train_batch(self, train_set, batch_size, tgtID_FullLabelMap):

    """
    :param train_set: tuples of positive trainpair in format of ([source_tokens, target_tokens, src_len, tgt_len, tgtID])
    :param batch_size: postive samples in each batch
    :param MAX_SEQ_LEN: max seq length for each input data.
    :param tgtID_FullLabelMap:  full targetSpace tgtID to label mapping
    :return: a tuple of batched mixed  samples in format of:
        (source_inputs, labels, src_length)
        source_inputs: [batch_size , MAX_SEQ_LEN]
        labels: [batch_size ] : integar ranged from [0, num_of_tgt_classes
        src_length: [batch_size ]
    """

    # Get a random batch of positive training pairs from train_set,
    # and then generate labels ( classes labeled from [0, num_target_space)

    source_inputs, labels, src_lens = [], [], []
    for idx in xrange(batch_size):
      #add a postive pair to batch
      source_input,  src_len,  tgtID = random.choice(train_set)
      source_inputs.append( source_input )
      labels.append( tgtID_FullLabelMap[tgtID] )
      src_lens.append( src_len )

    return  np.array(source_inputs, dtype=np.int32), np.array(labels, dtype=np.int64), np.array(src_lens, dtype=np.int32)


  def get_predict_feed_dict(self, srcSeqs, tgtSeqs, src_lens, tgt_lens ):
    """
    Returns a batch feed dict for given srcSequences passed as
    [batch_size, srcSequenceTokenIds].

    """
    d = {}
    d[self._src_input_data] = np.array(srcSeqs, dtype=np.int32)
    d[self._tgt_input_data] = np.array(tgtSeqs, dtype=np.int32)
    d[self._src_lens] = np.array(src_lens, dtype=np.int32)
    d[self._tgt_lens] = np.array(tgt_lens, dtype=np.int32)
    return d


  def get_train_feed_dict(self, srcSeqs, tgtSeqs, labels, src_lens, tgt_lens):
    """
    Returns a batch feed dict for given srcSquence and tgtSequences.

    """
    d = {}
    d[self._src_input_data] = np.array(srcSeqs, dtype=np.int32)
    d[self._labels] = np.array(labels, dtype=np.int64)
    d[self._tgt_input_data] = np.array(tgtSeqs, dtype=np.int32)
    d[self._src_lens] = np.array(src_lens, dtype=np.int32)
    d[self._tgt_lens] = np.array(tgt_lens, dtype=np.int32)
    return d
