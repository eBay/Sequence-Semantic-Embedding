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
import os
import codecs
import numpy as np
import tensorflow as tf
import data_utils
import sse_model


from flask import Flask, request, jsonify


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
tf.app.flags.DEFINE_integer("max_epoc", 8, "max epoc number for training procedure.")
tf.app.flags.DEFINE_integer("predict_nbest", 20, "max top N for evaluation prediction.")
tf.app.flags.DEFINE_string("data_dir", 'rawdata', "Data directory")
tf.app.flags.DEFINE_string("model_dir", 'models', "Trained model directory.")
tf.app.flags.DEFINE_string("export_dir", 'exports', "Trained model directory.")
tf.app.flags.DEFINE_string("device", "gpu:0", "Default to use GPU:0. Softplacement used, if no GPU found, further default to cpu:0.")
tf.app.flags.DEFINE_string("network_mode", 'source-encoder-only',
                            "Setup SSE network configration mode. SSE support three types of modes: source-encoder-only, dual-encoder, shared-encoder.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
FLAGS = tf.app.flags.FLAGS


def load_encodedTargetSpace():
  # load target sequence vocabulary file first
  tgtID_EncodingMap={}
  tgtID_FullLableMap={}
  fullLabel_tgtID_Map=[]
  line_cnt=0
  target_inputs = []
  target_lens= []
  for line in codecs.open( 'models/encoded.FullTargetSpace', 'r', 'utf-8'):
    tgtID, encodingIDs = line.strip().split('\t')
    tgt_tokens = [ int(tokenid) for tokenid in encodingIDs.split(' ')]
    tgt_len = len(tgt_tokens)
    tgtID_EncodingMap[tgtID] = tgt_tokens
    tgtID_FullLableMap[tgtID] = line_cnt
    fullLabel_tgtID_Map.append(tgtID)
    tgt_tokens = tgt_tokens + [data_utils.PAD_ID] * ( FLAGS.max_seq_length - tgt_len)
    target_inputs.append(tgt_tokens)
    target_lens.append( tgt_len )
    line_cnt += 1
  #load target ID Name mapping
  tgtID_Name_Map={}
  for line in codecs.open( 'models/targetIDs', 'r', 'utf-8'):
    tgtName, tgtID = line.strip().split('\t')
    tgtID_Name_Map[tgtID] = tgtName
  #Debug
  print("example of #1 TargetSpace record [tgtID, tgtEncodingMap, tgtFullSpaceLable, tgt_input, tgt_len]. \n [%s, %s, %s, %s, %s]" %
          (str(fullLabel_tgtID_Map[1]), str( tgtID_EncodingMap[ fullLabel_tgtID_Map[1] ]), str(1), target_inputs[1], target_lens[1] ))

  return tgtID_Name_Map, tgtID_EncodingMap, tgtID_FullLableMap, fullLabel_tgtID_Map,  \
         np.array(target_inputs,dtype=np.int32), np.array( target_lens, dtype=np.int32)


class FlaskApp(Flask):

  def __init__(self, *args, **kwargs):
    super(FlaskApp, self).__init__(*args, **kwargs)

    self.catreco_model = 'Do my initialization work here'

    if not os.path.exists(FLAGS.model_dir):
      print('Model folder does not exist!!')
      exit(-1)

    encodedFullTargetSpace_path = os.path.join(FLAGS.model_dir, "encoded.FullTargetSpace")

    if not os.path.exists(encodedFullTargetSpace_path):
      print('Encoded full target space file not exist!!')
      exit(-1)

    # load full set targetSeqID data
    #tgtID_Name_Map, tgtID_EncodingMap, tgtID_FullLableMap, fullLabel_tgtID_Map, target_inputs, target_lens = load_encodedTargetSpace(encodedFullTargetSpace_path)
    self.tgtID_Name_Map, self.tgtID_EncodingMap, self.tgtID_FullLableMap, self.fullLabel_tgtID_Map, self.target_inputs, self.target_lens = load_encodedTargetSpace()

    cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    self.sess = tf.Session(config=cfg)
    # check and load tensorflow models and related target space files
    self.model = sse_model.SSEModel(FLAGS.max_seq_length, FLAGS.max_gradient_norm, FLAGS.src_vocab_size,
                                  FLAGS.tgt_vocab_size,
                                  FLAGS.embedding_size, FLAGS.encoding_size,
                                  FLAGS.src_cell_size, FLAGS.tgt_cell_size, FLAGS.num_layers,
                                  FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, len(self.tgtID_FullLableMap),
                                  network_mode=FLAGS.network_mode, forward_only=True,
                                  TOP_N=FLAGS.predict_nbest)  # Load vocabularies.
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
      print('Error!!!Could not load model from specified folder: %s' % FLAGS.model_dir)
      self.sess.close()
      exit(-1)
    src_vocab_path = os.path.join(FLAGS.model_dir, "vocab.src")
    tgt_vocab_path = os.path.join(FLAGS.model_dir, "vocab.tgt")
    self.src_vocab, _ = data_utils.initialize_vocabulary(src_vocab_path)
    _, self.rev_tgt_vocab = data_utils.initialize_vocabulary(tgt_vocab_path)


app = FlaskApp(__name__)


@app.route('/api/catreco', methods=['GET'])
def catreco():
    #parse out GET request parameters: e.g.:  /api/catreco?title=iphone5s&?nbest=10
    title = request.args.get('title')
    if 'nbest' in request.args:
      nbest = int(request.args.get('nbest'))
    else:
      nbest = 10

    # inference tensorflow model
    # Get token-ids for the input sentence.
    source_tokens = data_utils.sentence_to_token_ids(tf.compat.as_bytes(title), app.src_vocab,
                                                     normalize_digits=True)
    src_len = len(source_tokens)
    if src_len > FLAGS.max_seq_length:
      source_tokens = source_tokens[:FLAGS.max_seq_length]
    else:
      source_tokens = source_tokens + [data_utils.PAD_ID] * (FLAGS.max_seq_length - src_len)

    dict = app.model.get_predict_feed_dict(np.array([source_tokens]), app.target_inputs, np.array([src_len]),
                                       app.target_lens)

    pred_conf, pred_labels = app.sess.run([app.model.predicted_tgts_score, app.model.predicted_labels], feed_dict=dict)
    pred_labels = np.vstack(pred_labels)
    pred_conf = np.vstack(pred_conf)
    top_confs = pred_conf[0][:nbest]
    top_tgtIDs = [app.fullLabel_tgtID_Map[lbl] for lbl in pred_labels[0][:nbest]]
    top_tgtNames = [app.tgtID_Name_Map[id] for id in top_tgtIDs]
    topCategories = []
    for idx in range(nbest):
        print('top%d:  %s , %f ,  %s ' % (idx + 1, top_tgtIDs[idx], top_confs[idx], top_tgtNames[idx]))
        entry={}
        entry['leafCatId'] = top_tgtIDs[idx]
        entry['leafCatName'] = top_tgtNames[idx]
        entry['confScore'] = float(top_confs[idx])
        topCategories.append(entry)
    return jsonify( { 'ReqeustTitle':title, 'ClassifyResults':topCategories} )


@app.route('/', methods=['GET'])
def default():
    return 'Default SSE Classification server. Please send GET request in the form of: /api/catreco?title=hello kitty sunglasses'

