# coding=utf-8
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from builtins import str
from builtins import str
from builtins import range
import math
import os, logging
import random
import sys
import time
import codecs

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import sse_model
import sse_evaluator
import sse_index
import text_encoder
from data import *

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")

tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training(positive pair count based).")
tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of word embedding vector.")
tf.app.flags.DEFINE_integer("encoding_size", 80, "Size of sequence encoding vector. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("src_cell_size", 96, "LSTM cell size in source RNN model.")
tf.app.flags.DEFINE_integer("tgt_cell_size", 96, "LSTM cell size in target RNN model. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 32000, "If no vocabulary file provided, will use this size to build vocabulary file from training data.")
tf.app.flags.DEFINE_integer("max_seq_length", 50, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("max_epoc", 10, "max epoc number for training procedure.")
tf.app.flags.DEFINE_integer("predict_nbest", 10, "max top N for evaluation prediction.")
tf.app.flags.DEFINE_string("task_type", 'classification',
                           "Type of tasks. We provide data, training receipe and service demos for four different type tasks:  classification, ranking, qna, crosslingual")

tf.app.flags.DEFINE_string("data_dir", 'rawdata-classification', "Data directory")

tf.app.flags.DEFINE_string("model_dir", 'models-classification', "Trained model directory.")
tf.app.flags.DEFINE_string("rawfilename", 'targetIDs', "raw target sequence file to be indexed")
tf.app.flags.DEFINE_string("encodedIndexFile", 'targetEncodingIndex.tsv', "target sequece encoding index file.")

tf.app.flags.DEFINE_string("device", "0", "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")
tf.app.flags.DEFINE_string("network_mode", 'dual-encoder',
                            "Setup SSE network configration mode. SSE support three types of modes: source-encoder-only, dual-encoder, shared-encoder.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device  # value can be 0,1,2, 3


def create_model(session, targetSpaceSize, vocabsize, forward_only):
  """Create SSE model and initialize or load parameters in session."""

  modelParams = {'max_seq_length': FLAGS.max_seq_length, 'vocab_size': vocabsize,
                 'embedding_size': FLAGS.embedding_size, 'encoding_size': FLAGS.encoding_size,
                 'learning_rate': FLAGS.learning_rate, 'learning_rate_decay_factor': FLAGS.learning_rate_decay_factor,
                 'src_cell_size':FLAGS.src_cell_size, 'tgt_cell_size':FLAGS.tgt_cell_size,
                 'network_mode': FLAGS.network_mode, 'predict_nbest':FLAGS.predict_nbest,
                 'targetSpaceSize':targetSpaceSize, 'forward_only': forward_only}

  data_utils.save_model_configs(FLAGS.model_dir, modelParams)

  model = sse_model.SSEModel( modelParams )

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    if forward_only:
      print('Error!!!Could not load any model from specified folder: %s' % FLAGS.model_dir )
      exit(-1)
    else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
  return model



def set_up_logging(run_id):
  formatter = logging.Formatter("%(asctime)s: %(message)s")
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.INFO)

  file_handler = logging.FileHandler("%s.log" % run_id)
  file_handler.setFormatter(formatter)
  root_logger.addHandler(file_handler)

  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(formatter)
  root_logger.addHandler(console_handler)


def train():
  # Prepare data.
  print("Preparing Train & Eval data in %s" % FLAGS.data_dir)

  for d in FLAGS.data_dir, FLAGS.model_dir:
    if not os.path.exists(d):
      os.makedirs(d)

  data = Data(FLAGS.model_dir,FLAGS.data_dir, FLAGS.vocab_size, FLAGS.max_seq_length)
  epoc_steps = len(data.rawTrainPosCorpus) /  FLAGS.batch_size

  print( "Training Data: %d total positive samples, each epoch need %d steps" % (len(data.rawTrainPosCorpus), epoc_steps ) )

  cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  with tf.Session(config=cfg) as sess:
    model = create_model( sess, data.rawnegSetLen, data.vocab_size, False )

    #setup tensorboard logging
    sw =  tf.summary.FileWriter( logdir=FLAGS.model_dir,  graph=sess.graph, flush_secs=120)
    summary_op = model.add_summaries()
    # This is the training loop.
    step_time, loss, train_acc = 0.0, 0.0, 0.0
    current_step = 0
    previous_accuracies = []
    for epoch in range( FLAGS.max_epoc ):
      epoc_start_Time = time.time()
      for batchId in range( epoc_steps ):
        start_time = time.time()
        source_inputs, tgt_inputs, labels = data.get_train_batch(FLAGS.batch_size)
        model.set_forward_only(False)
        d = model.get_train_feed_dict(source_inputs, tgt_inputs, labels)
        ops = [model.train, summary_op, model.loss, model.train_acc ]
        _, summary, step_loss, step_train_acc = sess.run(ops, feed_dict=d)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        train_acc += step_train_acc / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          print ("global epoc: %.3f, global step %d, learning rate %.4f step-time:%.2f loss:%.4f train_binary_acc:%.4f " %
                 ( float(model.global_step.eval())/ float(epoc_steps), model.global_step.eval(), model.learning_rate.eval(),
                             step_time, step_loss, train_acc ))
          checkpoint_path = os.path.join(FLAGS.model_dir, "SSE-LSTM.ckpt")
          acc_sum = tf.Summary(value=[tf.Summary.Value(tag="train_binary_acc", simple_value=train_acc)])
          sw.add_summary(acc_sum, current_step)

          # #########debugging##########
          # model.set_forward_only(True)
          # sse_index.createIndexFile(model, encoder, os.path.join(FLAGS.model_dir, FLAGS.rawfilename),
          #                           FLAGS.max_seq_length, os.path.join(FLAGS.model_dir, FLAGS.encodedIndexFile), sess,
          #                           batchsize=1000)
          # evaluator = sse_evaluator.Evaluator(model, eval_corpus, os.path.join(FLAGS.model_dir, FLAGS.encodedIndexFile),
          #                                     sess)
          # acc1, acc3, acc10 = evaluator.eval()
          # print("epoc# %.3f, task specific evaluation: top 1/3/10 accuracies: %f / %f / %f " % (float(model.global_step.eval())/ float(epoc_steps), acc1, acc3, acc10))
          # ###end of debugging########

          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_accuracies) > 3 and train_acc < min(previous_accuracies[-2:]):
            sess.run(model.learning_rate_decay_op)
          previous_accuracies.append(train_acc)
          # save currently best-ever model
          if train_acc == max(previous_accuracies):
            print("Better Accuracy %.4f found. Saving current best model ..." % train_acc )
            model.save(sess, checkpoint_path + "-BestEver")
          else:
            print("Best Accuracy is: %.4f, while current round is: %.4f" % (max(previous_accuracies), train_acc) )
            print("skip saving model ...")
          # if finished at least 2 Epocs and still no further accuracy improvement, stop training
          # report the best accuracy number and final model's number and save it.
          if epoch > 10 and train_acc < min(previous_accuracies[-5:]):
            p = model.save(sess, checkpoint_path + "-final")
            print("After around %d Epocs no further improvement, Training finished, wrote checkpoint to %s." % (epoch, p) )
            break

          # reset current checkpoint step statistics
          step_time, loss, train_acc = 0.0, 0.0, 0.0


      epoc_train_time = time.time() - epoc_start_Time
      print('\n\n\nepoch# %d  took %f hours' % ( epoch , epoc_train_time / (60.0 * 60) ) )

      # run task specific evaluation afer each epoch
      if (FLAGS.task_type not in ['ranking', 'crosslingual']) or ( (epoch+1) % 20 == 0 ):
        model.set_forward_only(True)
        sse_index.createIndexFile( model, data.encoder, os.path.join(FLAGS.model_dir, FLAGS.rawfilename), FLAGS.max_seq_length, os.path.join(FLAGS.model_dir, FLAGS.encodedIndexFile), sess, batchsize=1000 )
        evaluator = sse_evaluator.Evaluator(model, data.rawEvalCorpus, os.path.join(FLAGS.model_dir, FLAGS.encodedIndexFile) , sess)
        acc1, acc3, acc10 = evaluator.eval()
        print("epoc#%d, task specific evaluation: top 1/3/10 accuracies: %f / %f / %f \n\n\n" % (epoch, acc1, acc3, acc10) )
      # Save checkpoint at end of each epoch
      checkpoint_path = os.path.join(FLAGS.model_dir, "SSE-LSTM.ckpt")
      model.save(sess, checkpoint_path + '-epoch-%d'%epoch)
      if len(previous_accuracies) > 0:
        print('So far best ever model training binary accuracy is: %.4f ' % max(previous_accuracies) )


def main(_):

  if not FLAGS.data_dir or not FLAGS.model_dir:
    print("--data_dir and --model_dir must be specified.")
    sys.exit(1)


  run_id = 'BatchSize' + str(FLAGS.batch_size)  + '.EmbedSize' + str(FLAGS.embedding_size) + \
            '.EncodeSize' + str(FLAGS.encoding_size) + '.SrcCell' + str(FLAGS.src_cell_size) + \
           '.TgtCell' + str(FLAGS.tgt_cell_size) + '.SrcCell' + str(FLAGS.src_cell_size) + \
           '.' + str(FLAGS.network_mode) + \
           '.' + str(time.time())[-5:]
  set_up_logging(run_id)
  train()

if __name__ == "__main__":
  tf.app.run()
