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



tf.app.flags.DEFINE_float("learning_rate", 0.3, "Learning rate.")
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
tf.app.flags.DEFINE_integer("src_vocab_size", 50000, "Source Sequence vocabulary size in the mapping task.")
tf.app.flags.DEFINE_integer("tgt_vocab_size", 3000, "Target Sequence vocabulary size in the mapping task.")

tf.app.flags.DEFINE_integer("max_seq_length", 55, "max number of words in each source or target sequence.")
tf.app.flags.DEFINE_integer("max_epoc", 5, "max epoc number for training procedure.")
tf.app.flags.DEFINE_integer("predict_nbest", 20, "max top N for evaluation prediction.")


tf.app.flags.DEFINE_string("data_dir", 'rawdata', "Data directory")
tf.app.flags.DEFINE_string("model_dir", 'models', "Trained model directory.")
tf.app.flags.DEFINE_string("export_dir", 'exports', "Trained model directory.")
tf.app.flags.DEFINE_string("device", "gpu:0", "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")

tf.app.flags.DEFINE_string("network_mode", 'source-encoder-only',
                            "Setup SSE network configration mode. SSE support three types of modes: source-encoder-only, dual-encoder, shared-encoder.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")


tf.app.flags.DEFINE_boolean("demo", False,
                            "Set to True for interactive demo of testing classification.")
tf.app.flags.DEFINE_boolean("embeddingMode", False,
                            "Set to True to generate embedding vectors file for entries in targetIDs file.")

FLAGS = tf.app.flags.FLAGS


def read_train_data(encoded_trainpair_path, encodedTargetSpaceMap):
  """Read data from source and target files.

  Args:
    encoded_trainpair_path: targetID, encoded source tokenIDs.
    encodedTargetSpaceMap: targetID and encoded target seqence tokenIDs.

  Returns:
    data_set: a list of positive (sourceTokenIDs, targetTokenIDs) pairs read from the provided data files.
    batchsize_per_epoc: how many batched needed to go through out one epoch.
  """
  data_set = []
  counter = 0
  for line in codecs.open(encoded_trainpair_path, "r", 'utf-8'):
    tgtID, srcSeq = line.strip().split('\t')
    counter += 1
    if counter % 1000000 == 0:
      print("  reading data line %d" % counter)
      sys.stdout.flush()
    #get source input sequence and PADDING accordingly
    source_tokens = [int(x) for x in srcSeq.split()]
    src_len = len(source_tokens)
    if src_len > FLAGS.max_seq_length:
      print('Error Deteced!!! Source input seq length is:%s. \n Excced current MAX_SEQ_LENTH of %s. Try to increase limit!!!!' %
                       (str(src_len), str(FLAGS.max_seq_length) ) )
      continue
    source_tokens = source_tokens + [data_utils.PAD_ID] * (FLAGS.max_seq_length - src_len)
    #get targetID input
    if tgtID not in set(encodedTargetSpaceMap.keys()):
      print('Error Detected!! trouble in finding targetID in target Space file!! %s' % tgtID )
      continue
    data_set.append([source_tokens, src_len, tgtID])
    #debug
    if counter == 100:
      print("example of #100 dataset record [src_seq, src_len, tgtID]. \n [ %s, %s, %s]" %
            ( str(source_tokens),  str(src_len), str(tgtID) ) )
  return data_set, int(math.floor(counter/FLAGS.batch_size))


def get_eval_set(encoded_eval_pair_path ):
  src_seqs, src_lens, tgtIDs = [], [], []
  for line in codecs.open( encoded_eval_pair_path,'r', "utf-8"):
    id, srcSeqTokens = line.strip().split('\t')
    source_ids = [int(x) for x in srcSeqTokens.split()]
    src_len = len(source_ids)
    if src_len > FLAGS.max_seq_length:
      print('Error Detected!!! EvalSet source input seq length:%s, excceed MAX_LENG of %s.!!!!'
                       % ( str(src_len), str(FLAGS.max_seq_length)) )
      continue
    src_lens.append( src_len )
    source_ids = source_ids + [data_utils.PAD_ID] * ( FLAGS.max_seq_length - src_len )
    src_seqs.append( source_ids )
    tgtIDs.append( id )

  #Debug
  print("example of #1 eval-set record [src_seq, src_len, tgtID]. \n [%s, %s, %s]" %
          (str(src_seqs[1]), str(src_lens[1]), str(tgtIDs[1])))

  return src_seqs, src_lens, tgtIDs


def create_model(session, targetSpaceSize, forward_only):
  """Create SSE model and initialize or load parameters in session."""
  model = sse_model.SSEModel( FLAGS.max_seq_length, FLAGS.max_gradient_norm,  FLAGS.src_vocab_size, FLAGS.tgt_vocab_size,
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



def load_targetIdNameMap():
  tgtID_Name_Map={}
  for line in codecs.open( os.path.join(FLAGS.model_dir, 'targetIDs'), 'r', 'utf-8'):
    tgtName, tgtID = line.strip().split('\t')
    tgtID_Name_Map[tgtID] = tgtName

  return tgtID_Name_Map


def load_encodedTargetSpace( encodedTargetSpace_path ):
  """ Load encoded full Targets Space file.

  Args:
      encodedTargetSpace_path: file contains encodedTargetSpaceFile.

  Returns:
      a tuple of maps:(tgtID_tgtEncoding_Map, tgtID_FullSpaceLabel_Map, FullSpaceLabel_tgtID_Map, tgt_inputs, tgt_lens )
  """
  # load target sequence vocabulary file first
  tgtID_EncodingMap={}
  tgtID_FullLableMap={}
  fullLabel_tgtID_Map=[]
  line_cnt=0
  target_inputs = []
  target_lens= []
  for line in codecs.open( encodedTargetSpace_path, 'r', 'utf-8'):
    tgtID, encodingIDs = line.strip().split('\t')
    tgt_tokens = [ int(tokenid) for tokenid in encodingIDs.split(' ')]
    tgt_len = len(tgt_tokens)
    if tgt_len > FLAGS.max_seq_length:
      print(
        'Error Detected!!! target input seq length is:%s. \n Excced current MAX_SEQ_LENTH of %s. Try to increase limit!!!!' %
        (str(tgt_len), str(FLAGS.max_seq_length)))
      continue
    tgtID_EncodingMap[tgtID] = tgt_tokens
    tgtID_FullLableMap[tgtID] = line_cnt
    fullLabel_tgtID_Map.append(tgtID)
    tgt_tokens = tgt_tokens + [data_utils.PAD_ID] * ( FLAGS.max_seq_length - tgt_len)
    target_inputs.append(tgt_tokens)
    target_lens.append( tgt_len )
    line_cnt += 1

  #Debug
  print("example of #1 TargetSpace record [tgtID, tgtEncodingMap, tgtFullSpaceLable, tgt_input, tgt_len]. \n [%s, %s, %s, %s, %s]" %
          (str(fullLabel_tgtID_Map[1]), str( tgtID_EncodingMap[ fullLabel_tgtID_Map[1] ]), str(1), target_inputs[1], target_lens[1] ))

  return tgtID_EncodingMap, tgtID_FullLableMap, fullLabel_tgtID_Map,  \
         np.array(target_inputs,dtype=np.int32), np.array( target_lens, dtype=np.int32)


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

  encoded_train_pair_path, encoded_eval_pair_path, encodedFullTargetSpace_path,  _, _  = data_utils.prepare_raw_data(
      FLAGS.data_dir, FLAGS.model_dir, FLAGS.src_vocab_size, FLAGS.tgt_vocab_size)

  #load full set targetSeqID data
  tgtID_EncodingMap, tgtID_FullLableMap, fullLabel_tgtID_Map, target_inputs, target_lens = load_encodedTargetSpace(encodedFullTargetSpace_path)

  #load full set train data
  print("Reading development and training data ..." )
  train_set, epoc_steps = read_train_data( encoded_train_pair_path, tgtID_EncodingMap )
  print("Training Data: %d total samples, each epoch need %d steps" % (len(train_set), epoc_steps ))

  #load eval data
  eval_src_seqs, eval_src_lens, eval_tgtIDs = get_eval_set( encoded_eval_pair_path )

  cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

  with tf.device('/' + FLAGS.device),  tf.Session(config=cfg) as sess:
    # Create SSE model and build tensorflow training graph.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.embedding_size))
    model = create_model( sess, len(tgtID_FullLableMap), False )

    #setup evaluation graph
    evaluator = sse_evaluator.Evaluator(model, eval_src_seqs, eval_src_lens, eval_tgtIDs, target_inputs, target_lens, tgtID_FullLableMap, sess)

    #setup tensorboard logging
    sw =  tf.summary.FileWriter( logdir=FLAGS.model_dir,  graph=sess.graph, flush_secs=120)
    summary_op = model.add_summaries()

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_accuracies = []
    for epoch in range( FLAGS.max_epoc ):
      epoc_start_Time = time.time()
      random.shuffle(train_set, random.random)
      for batchId in range(epoc_steps - int( 2.5 * FLAGS.steps_per_checkpoint) ): #basic drop out here
        start_time = time.time()
        source_inputs, labels, src_lens = [], [], []
        for idx in range(FLAGS.batch_size):
          source_input, src_len, tgtID = train_set[batchId * FLAGS.batch_size + idx]
          source_inputs.append(source_input)
          labels.append(tgtID_FullLableMap[tgtID])
          src_lens.append(src_len)

        d = model.get_train_feed_dict(source_inputs, target_inputs, labels, src_lens, target_lens)
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
    print( 'Encoded full target space file not exist!!')
    exit(-1)

  #load full set targetSeqID data
  tgtID_EncodingMap, tgtID_FullLableMap, fullLabel_tgtID_Map, target_inputs, target_lens = load_encodedTargetSpace(encodedFullTargetSpace_path)
  tgtID_Name_Map = load_targetIdNameMap()

  cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  with tf.device('/' + FLAGS.device),  tf.Session(config=cfg) as sess:
    # TODO: improve here later
    # load tensorflow model from model_dir's checkpoint filer
    model = create_model(sess, len(tgtID_FullLableMap), True)

    # Load vocabularies.
    # TODO: rename vocab file to more consistant names
    src_vocab_path = os.path.join(FLAGS.model_dir, "vocab.src" )
    tgt_vocab_path = os.path.join(FLAGS.model_dir, "vocab.tgt" )
    src_vocab, _ = data_utils.initialize_vocabulary(src_vocab_path)
    _, rev_tgt_vocab = data_utils.initialize_vocabulary(tgt_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence and sentence.strip().lower() != 'exit':
      # Get token-ids for the input sentence.
      source_tokens = data_utils.sentence_to_token_ids( tf.compat.as_str(sentence), src_vocab, normalize_digits=True)
      src_len = len(source_tokens)
      if src_len > FLAGS.max_seq_length:
        print(
          'Error Deteced!!!  input seq length is:%s. \n Excced current MAX_SEQ_LENTH of %s. Try to increase limit!!!!' %
          (str(src_len), str( FLAGS.max_seq_length )))
        continue
      source_tokens = source_tokens + [data_utils.PAD_ID] * ( FLAGS.max_seq_length - src_len)
      dict = model.get_predict_feed_dict( np.array([source_tokens]), target_inputs, np.array([src_len]), target_lens)
      pred_conf, pred_labels = sess.run( [model.predicted_tgts_score, model.predicted_labels], feed_dict=dict)
      pred_labels = np.vstack(pred_labels)
      pred_conf = np.vstack(pred_conf)
      top5_confs = pred_conf[0][:5]
      top5_tgtIDs = [ fullLabel_tgtID_Map[lbl] for lbl in pred_labels[0][:5]]
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
