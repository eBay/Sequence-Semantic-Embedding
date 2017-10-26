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
import text_encoder

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training(positive pair count based).")
tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of word embedding vector.")
tf.app.flags.DEFINE_integer("encoding_size", 80, "Size of sequence encoding vector. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("src_cell_size", 96, "LSTM cell size in source RNN model.")
tf.app.flags.DEFINE_integer("tgt_cell_size", 96, "LSTM cell size in target RNN model. Same number of nodes for each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 32000, "If no vocabulary file provided, will use this size to build vocabulary file from training data.")
tf.app.flags.DEFINE_integer("max_seq_length", 100, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("max_epoc", 8, "max epoc number for training procedure.")
tf.app.flags.DEFINE_integer("predict_nbest", 20, "max top N for evaluation prediction.")

tf.app.flags.DEFINE_string("task_type", 'classification',
                           "Type of tasks. We provide data, training receipe and service demos for four different type tasks:  classification, ranking, qna, crosslingual")
tf.app.flags.DEFINE_string("data_dir", 'rawdata-classification', "Data directory")
tf.app.flags.DEFINE_string("model_dir", 'models-classification', "Trained model directory.")
tf.app.flags.DEFINE_string("export_dir", 'exports-classification', "Trained model directory.")
tf.app.flags.DEFINE_string("device", "0", "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")
tf.app.flags.DEFINE_string("network_mode", 'dual-encoder',
                            "Setup SSE network configration mode. SSE support three types of modes: source-encoder-only, dual-encoder, shared-encoder.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("demo", False,
                            "Set to True for interactive demo of testing classification.")
tf.app.flags.DEFINE_boolean("embeddingMode", False,
                            "Set to True to generate embedding vectors file for entries in targetIDs file.")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device  # value can be 0,1,2, 3


def create_model(session, targetSpaceSize, vocabsize, forward_only):
  """Create SSE model and initialize or load parameters in session."""
  modelConfigs = ( FLAGS.max_seq_length, FLAGS.max_gradient_norm,  vocabsize,
      FLAGS.embedding_size, FLAGS.encoding_size,
      FLAGS.src_cell_size, FLAGS.tgt_cell_size, FLAGS.num_layers,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, targetSpaceSize ,
      FLAGS.network_mode , FLAGS.predict_nbest )

  data_utils.save_model_configs(FLAGS.model_dir, modelConfigs)

  model = sse_model.SSEModel( FLAGS.max_seq_length, FLAGS.max_gradient_norm,  vocabsize,
      FLAGS.embedding_size, FLAGS.encoding_size,
      FLAGS.src_cell_size, FLAGS.tgt_cell_size, FLAGS.num_layers,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, targetSpaceSize ,
      network_mode=FLAGS.network_mode , forward_only=forward_only, TOP_N=FLAGS.predict_nbest )

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

  encoder, train_corpus, dev_corpus, encodedTgtSpace, tgtIdNameMap = data_utils.prepare_raw_data(
      FLAGS.data_dir, FLAGS.model_dir, FLAGS.vocab_size , FLAGS.task_type, FLAGS.max_seq_length )

  epoc_steps = int(math.floor(len(train_corpus) / FLAGS.batch_size))

  print( "Training Data: %d total samples (pos + neg), each epoch need %d steps" % (len(train_corpus), epoc_steps ) )

  cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

  with tf.Session(config=cfg) as sess:
    # Create SSE model and build tensorflow training graph.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.embedding_size))
    model = create_model( sess, len(encodedTgtSpace), encoder.vocab_size,  False )

    #setup evaluation graph
    evaluator = sse_evaluator.Evaluator(model, dev_corpus, encodedTgtSpace , sess)

    #setup tensorboard logging
    sw =  tf.summary.FileWriter( logdir=FLAGS.model_dir,  graph=sess.graph, flush_secs=120)
    summary_op = model.add_summaries()

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_accuracies = []
    fullSetTargetIds = set(encodedTgtSpace.keys())
    for epoch in range( FLAGS.max_epoc ):
      epoc_start_Time = time.time()
      random.shuffle(train_corpus, random.random)
      for batchId in range(epoc_steps - int( 2.5 * FLAGS.steps_per_checkpoint) ): #basic drop out here
        start_time = time.time()
        source_inputs, src_lens, tgt_inputs, tgt_lens, labels  = [], [], [], [], []
        for idx in range(FLAGS.batch_size):
          source_input, tgtId = train_corpus[batchId * FLAGS.batch_size + idx]
          #add positive pair
          source_inputs.append(source_input)
          src_lens.append(source_input.index(text_encoder.PAD_ID) +1)
          tgt_inputs.append(encodedTgtSpace[tgtId])
          tgt_lens.append(encodedTgtSpace[tgtId].index(text_encoder.PAD_ID) +1)
          labels.append(1.0)
          #add negative pair
          negTgt = random.sample( fullSetTargetIds - set([tgtId]) , 1)[0]
          source_inputs.append(source_input)
          src_lens.append(source_input.index(text_encoder.PAD_ID) +1)
          tgt_inputs.append(encodedTgtSpace[negTgt])
          tgt_lens.append(encodedTgtSpace[negTgt].index(text_encoder.PAD_ID) +1)
          labels.append(0.0)

        d = model.get_train_feed_dict(source_inputs, tgt_inputs, labels, src_lens, tgt_lens)
        ops = [model.train, summary_op, model.loss]
        _, summary, step_loss = sess.run(ops, feed_dict=d)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          print ("global epoc: %.3f, global step %d, learning rate %.4f step-time:%.2f loss:%.4f " %
                 ( float(model.global_step.eval())/ float(epoc_steps), model.global_step.eval(), model.learning_rate.eval(),
                             step_time, step_loss ))
          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(FLAGS.model_dir, "SSE-LSTM.ckpt")
          # model.save(sess, checkpoint_path, global_step=model.global_step)  #only save better models
          step_time, loss = 0.0, 0.0
          # Run evals on development set and print their accuracy number.
          t = time.time()
          acc1, acc3, acc10 = evaluator.eval()
          acc_sum = tf.Summary(value=[tf.Summary.Value(tag="acc1", simple_value=acc1),
                                        tf.Summary.Value(tag="acc3", simple_value=acc3),
                                        tf.Summary.Value(tag="acc10", simple_value=acc10)])
          sw.add_summary(acc_sum, current_step)
          print("Step %d, top 1/3/10 accuracies: %f / %f / %f, (eval took %f seconds) " %
                (current_step,  acc1, acc3, acc10, time.time() - t) )
          sys.stdout.flush()
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_accuracies) > 2 and acc1 < min(previous_accuracies[-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_accuracies.append(acc1)
          # save currently best-ever model
          if acc1 == max(previous_accuracies):
            print("Better Accuracy %f found. Saving current best model ..." % acc1 )
            model.save(sess, checkpoint_path + "-BestEver")
          else:
            print("Best Accuracy is: %f, while current round is: %f" % (max(previous_accuracies), acc1) )
            print("skip saving model ...")
          # if finished at least 2 Epocs and still no further accuracy improvement, stop training
          # report the best accuracy number and final model's number and save it.
          if epoch > 2 and acc1 < min(previous_accuracies[-3:]):
            p = model.save(sess, checkpoint_path + "-final")
            print("After around %d Epocs no further improvement, Training finished, wrote checkpoint to %s." % (epoch, p) )
            print("Best ever top1 accuracy: %.2f , Final top 1 / 3 / 10 accuracies: %.2f / %.2f / %.2f" %
                  (max(previous_accuracies), acc1, acc3, acc10 ) )
            break
      #give out epoc statistics
      epoc_train_time = time.time() - epoc_start_Time
      print('epoch# %d  took %f hours' % ( epoch , epoc_train_time / (60.0 * 60) ) )
      # Save checkpoint at end of each epoch
      checkpoint_path = os.path.join(FLAGS.model_dir, "SSE-LSTM.ckpt")
      model.save(sess, checkpoint_path + '-epoch-%d'%epoch)
      if len(previous_accuracies) > 0:
        print('So far best ever model top1 accuracy is: %.4f ' % max(previous_accuracies) )


def demo():
  if not os.path.exists( FLAGS.model_dir ):
    print('Model folder does not exist!!')
    exit(-1)
  encodedFullTargetSpace_path = os.path.join(FLAGS.model_dir, "encoded.FullTargetSpace")
  if not os.path.exists( encodedFullTargetSpace_path):
    print( 'Encoded full target space file not exist. Please ReTrain the model to get it!!')
    exit(-1)

  #load full set targetSeqID data
  encoder, encodedTgtSpace, tgtID_Name_Map =  data_utils.load_encodedTargetSpace(FLAGS.model_dir)
  fullTgtIdList = encodedTgtSpace.keys()
  tgtLabel_IDMap = {idx: tgtid for (idx, tgtid) in enumerate(fullTgtIdList)}
  tgtInput_batches = [encodedTgtSpace[tgtid] for tgtid in fullTgtIdList]
  tgtLen_batches = [encodedTgtSpace[tgtid].index(text_encoder.PAD_ID) + 1 for tgtid in fullTgtIdList]

  cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  with tf.Session(config=cfg) as sess:
    # TODO: improve here later
    #load model
    modelConfigs = data_utils.load_model_configs(FLAGS.model_dir)
    model = sse_model.SSEModel( int(modelConfigs['max_seq_length']), float(modelConfigs['max_gradient_norm']), int(modelConfigs['vocabsize']),
                               int(modelConfigs['embedding_size']), int(modelConfigs['encoding_size']),
                               int(modelConfigs['src_cell_size']), int(modelConfigs['tgt_cell_size']), int(modelConfigs['num_layers']),
                               float(modelConfigs['learning_rate']), float(modelConfigs['learning_rate_decay_factor']), int(modelConfigs['targetSpaceSize']), network_mode=modelConfigs['network_mode'], forward_only=True, TOP_N=int(modelConfigs['TOP_N']) )
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt:
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Error!!!Could not load any model from specified folder: %s' % FLAGS.model_dir)
        exit(-1)

    # Decode from standard input.
    sys.stdout.write("\n\nPlease type some keywords to get related task results.\nType 'exit' to quit demo.\n > ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence and sentence.strip().lower() != 'exit':
      # Get token-ids for the input sentence.
      source_tokens = encoder.encode( tf.compat.as_str(sentence).lower())
      srclen = len(source_tokens)
      if srclen > int(modelConfigs['max_seq_length']) - 1:
        print(
          'Max number of supported keywords is  %d \n Please try againt!!!!' % ( int(modelConfigs['max_seq_length']) ) )
        continue
      source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * ( int(modelConfigs['max_seq_length']) - srclen - 1)

      print("")

      dict = model.get_predict_feed_dict( np.array([source_tokens]), tgtInput_batches, np.array([srclen]), tgtLen_batches)
      pred_conf, pred_labels = sess.run( [model.predicted_tgts_score, model.predicted_labels], feed_dict=dict)
      pred_labels = np.vstack(pred_labels)
      pred_conf = np.vstack(pred_conf)
      top5_confs = pred_conf[0][:5]
      top5_tgtIDs = [ tgtLabel_IDMap[lbl] for lbl in pred_labels[0][:5]]
      top5_tgtNames = [ tgtID_Name_Map[id] for id in top5_tgtIDs ]

      print('Top 5 Prediction results are:\n')
      for idx in range(5):
        print( 'top%d:  %s , %f ,  %s ' % ( idx+1, top5_tgtIDs[idx], top5_confs[idx], top5_tgtNames[idx]) )
      print("> ", end="")

      sys.stdout.flush()
      sentence = sys.stdin.readline()



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

  if FLAGS.demo:
    demo()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
