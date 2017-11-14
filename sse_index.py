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
# @Date:   2017-10-24
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
import sys
import codecs

import numpy as np
import tensorflow as tf

import data_utils
import sse_model
import text_encoder
import math

tf.app.flags.DEFINE_string("idx_model_dir", 'models-classification', "Trained model directory.")
tf.app.flags.DEFINE_string("idx_rawfilename", 'targetIDs', "raw target sequence file to be indexed")
tf.app.flags.DEFINE_string("idx_encodedIndexFile", 'targetEncodingIndex.tsv', "target sequece encoding index file.")


FLAGS = tf.app.flags.FLAGS


def createIndexFile( model, encoder, rawfile, max_seq_len, encodeIndexFile, session, batchsize=10000 ):

  if not os.path.exists(rawfile):
    print('Error!! Could not find raw target file to be indexed!! :%s' % rawfile)
    exit(-1)

  # start to indexing
  outFile = codecs.open(encodeIndexFile, 'w', 'utf-8')
  rawdata = codecs.open(rawfile, 'r', 'utf-8').readlines()
  cnt = 0
  print("Start indexing whole target space entries with current model ...")
  for batchId in range(math.ceil(len(rawdata) / batchsize)):
    tgtInputs, tgtLens, tgtIds, tgtSentences = [], [], [], []
    for line in rawdata[batchId * batchsize:(batchId + 1) * batchsize]:
      cnt += 1
      # Get token-ids for the raw target sequence
      info = line.strip().split('\t')
      if len(info) != 2:
        print("Missing field with error line in raw target file: %s " % line)
        continue
      tgtSentence, tgtId = info[0], info[1]
      tgt_tokens = encoder.encode(tgtSentence.lower())
      tgtlen = len(tgt_tokens)
      if tgtlen > max_seq_len - 1:
        #print('Current raw tgt file line exceed max number of supported keywords(%d): %s!!!' % (max_seq_len, line))
        continue
      tgt_tokens = tgt_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * (max_seq_len - tgtlen - 1)
      tgtInputs.append(tgt_tokens)
      tgtLens.append(tgt_tokens.index(text_encoder.PAD_ID) + 1)
      tgtIds.append(tgtId)
      tgtSentences.append(tgtSentence)
    dict = model.get_target_encoding_feed_dict(tgtInputs, tgtLens)
    targetsEncodings = session.run([model.tgt_seq_embedding], feed_dict=dict)
    #targetsEncodings = session.run([model.norm_tgt_seq_embedding], feed_dict=dict)
    targetsEncodings = np.vstack(targetsEncodings)
    for idx in range(len(tgtSentences)):
      outFile.write(
        tgtIds[idx] + '\t' + tgtSentences[idx] + '\t' + ','.join([str(n) for n in targetsEncodings[idx]]) + '\n')
  print("Done of all indexing total count:%d" % cnt)
  outFile.close()


def index(model_dir, rawfile, encodeIndexFile, batchsize=10000):
  if not os.path.exists( model_dir ):
    print('Error! Model folder does not exist!! : %s' % model_dir)
    exit(-1)

  if not os.path.exists( os.path.join(model_dir, 'vocabulary.txt' ) ):
    print('Error!! Could not find vocabulary file for encoder in folder :%s' % model_dir)
    exit(-1)

  encoder = text_encoder.SubwordTextEncoder(filename=os.path.join(model_dir, 'vocabulary.txt' ))
  print("Loaded  vocab size is: %d" % encoder.vocab_size)

  cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  with tf.Session(config=cfg) as sess:
    #load model
    modelConfigs = data_utils.load_model_configs(model_dir)
    model = sse_model.SSEModel( int(modelConfigs['max_seq_length']), float(modelConfigs['max_gradient_norm']), int(modelConfigs['vocabsize']),
                               int(modelConfigs['embedding_size']), int(modelConfigs['encoding_size']),
                               int(modelConfigs['src_cell_size']), int(modelConfigs['tgt_cell_size']), int(modelConfigs['num_layers']),
                               float(modelConfigs['learning_rate']), float(modelConfigs['learning_rate_decay_factor']), int(modelConfigs['targetSpaceSize']), network_mode=modelConfigs['network_mode'], forward_only=True, TOP_N=int(modelConfigs['TOP_N']) )
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt:
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Error!!!Could not load any model from specified folder: %s' % model_dir)
        exit(-1)

    # start to indexing
    createIndexFile(model, encoder, rawfile, int(modelConfigs['max_seq_length']), encodeIndexFile, sess, batchsize)



def main(_):
  if not FLAGS.idx_model_dir:
    print("--idx_model_dir must be specified.")
    sys.exit(1)
  index( FLAGS.idx_model_dir, os.path.join(FLAGS.idx_model_dir,FLAGS.idx_rawfilename) , os.path.join(FLAGS.idx_model_dir, FLAGS.idx_encodedIndexFile) )


if __name__ == "__main__":
  tf.app.run()
