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
import os
import codecs
import numpy as np
import tensorflow as tf
import data_utils
import sse_model
import text_encoder

from flask import Flask, request, jsonify

class FlaskApp(Flask):

  def __init__(self, *args, **kwargs):
    super(FlaskApp, self).__init__(*args, **kwargs)

    self.model = 'Do my initialization work here, loading model and index ....'
    self.model_type = os.environ.get("MODEL_TYPE", "classification")
    self.model_dir = "models-" + self.model_type
    self.indexFile = os.environ.get("INDEX_FILE", "targetEncodingIndex.tsv")
    print("In app class: Received flask appconfig is: " + os.environ.get('MODEL_TYPE', 'Default_classification') )

    if not os.path.exists(self.model_dir):
      print('Model folder %s does not exist!!' % self.model_dir )
      exit(-1)

    if not os.path.exists(os.path.join(self.model_dir, self.indexFile)):
      print('Index File does not exist!!')
      exit(-1)

    # load full set targetSeqID data
    if not os.path.exists(os.path.join(self.model_dir, 'vocabulary.txt')):
        print('Error!! Could not find vocabulary file for encoder in model folder.')
        exit(-1)
    self.encoder = text_encoder.SubwordTextEncoder(filename=os.path.join(self.model_dir, 'vocabulary.txt'))

    # load full set target Index data
    self.targetEncodings = []
    self.targetIDs = []
    self.targetIDNameMap = {}
    idx = 0
    for line in codecs.open(os.path.join(self.model_dir, self.indexFile), 'r', 'utf-8').readlines():
        info = line.strip().split('\t')
        if len(info) != 3:
            print('Error in targetIndexFile! %s' % line)
            continue
        tgtid, tgtseq, tgtEncoding = info[0], info[1], info[2]
        self.targetIDs.append(tgtid)
        self.targetEncodings.append([float(f) for f in tgtEncoding.strip().split(',')])
        self.targetIDNameMap[tgtid] = tgtseq
        idx += 1
    self.targetEncodings = np.array(self.targetEncodings)

    cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    self.sess = tf.Session(config=cfg)
    #load model
    self.modelConfigs = data_utils.load_model_configs(self.model_dir)
    self.model = sse_model.SSEModel( int(self.modelConfigs['max_seq_length']), float(self.modelConfigs['max_gradient_norm']),
                                     int(self.modelConfigs['vocabsize']),
                               int(self.modelConfigs['embedding_size']), int(self.modelConfigs['encoding_size']),
                               int(self.modelConfigs['src_cell_size']), int(self.modelConfigs['tgt_cell_size']), int(self.modelConfigs['num_layers']),
                               float(self.modelConfigs['learning_rate']), float(self.modelConfigs['learning_rate_decay_factor']),
                                     int(self.modelConfigs['targetSpaceSize']), network_mode=self.modelConfigs['network_mode'],
                                     forward_only=True, TOP_N=int(self.modelConfigs['TOP_N']) )
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt:
      print("loading model from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
        print('Error!!!Could not load any model from specified folder: %s' % self.model_dir)
        exit(-1)


app = FlaskApp(__name__)


@app.route('/api/classify', methods=['GET'])
def classification():
    #parse out classification task GET request parameters: e.g.:  /api/classify?keywords=hello kitty sunglasses&?nbest=8
    keywords = request.args.get('keywords')
    if 'nbest' in request.args:
      nbest = int(request.args.get('nbest'))
    else:
      nbest = 8

    # inference tensorflow model
    # Get token-ids for the input sentence.
    source_tokens = app.encoder.encode(tf.compat.as_str(keywords).lower())
    src_len = len(source_tokens)
    if src_len > int(app.modelConfigs['max_seq_length']):
      source_tokens = source_tokens[:int(app.modelConfigs['max_seq_length'])]
    else:
      source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * ( int(app.modelConfigs['max_seq_length']) - src_len - 1)
    dict = app.model.get_source_encoding_feed_dict(np.array([source_tokens]), np.array([src_len+1]))
    #sourceEncodings = app.sess.run([app.model.src_seq_embedding], feed_dict=dict)
    sourceEncodings = app.sess.run([app.model.norm_src_seq_embedding], feed_dict=dict)
    sourceEncodings = np.vstack(sourceEncodings)
    distances = np.dot(sourceEncodings, app.targetEncodings.T)
    rankedScore, rankedIdx = data_utils.getSortedResults(distances)
    top_confs = rankedScore[0][:nbest]
    top_tgtIDs = [app.targetIDs[lbl] for lbl in rankedIdx[0][:nbest]]
    top_tgtNames = [app.targetIDNameMap[id] for id in top_tgtIDs]
    topResults = []

    for idx in range(nbest):
        print('top%d:  %s , %f ,  %s ' % (idx + 1, top_tgtIDs[idx], top_confs[idx], top_tgtNames[idx]))
        entry={}
        entry['targetCategoryId'] = top_tgtIDs[idx]
        entry['targetCategoryName'] = top_tgtNames[idx]
        entry['confidenceScore'] = float(top_confs[idx])
        topResults.append(entry)
    return jsonify( { 'ReqeustKeywords':keywords, 'ClassificationResults':topResults} )

@app.route('/api/search', methods=['GET'])
def relevanceRanking():
    #parse out search ranking task GET request parameters: e.g.:  /api/search?query=red nike shoes&?nbest=10
    keywords = request.args.get('query')
    if 'nbest' in request.args:
      nbest = int(request.args.get('nbest'))
    else:
      nbest = 10

    # inference tensorflow model
    # Get token-ids for the input sentence.
    source_tokens = app.encoder.encode(tf.compat.as_str(keywords).lower())
    src_len = len(source_tokens)
    if src_len > int(app.modelConfigs['max_seq_length']):
      source_tokens = source_tokens[:int(app.modelConfigs['max_seq_length'])]
    else:
      source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * ( int(app.modelConfigs['max_seq_length']) - src_len - 1)
    dict = app.model.get_source_encoding_feed_dict(np.array([source_tokens]), np.array([src_len+1]))
    sourceEncodings = app.sess.run([app.model.src_seq_embedding], feed_dict=dict)
    #sourceEncodings = app.sess.run([app.model.norm_src_seq_embedding], feed_dict=dict)
    sourceEncodings = np.vstack(sourceEncodings)
    distances = np.dot(sourceEncodings, app.targetEncodings.T)
    rankedScore, rankedIdx = data_utils.getSortedResults(distances)
    top_confs = rankedScore[0][:nbest]
    top_tgtIDs = [app.targetIDs[lbl] for lbl in rankedIdx[0][:nbest]]
    top_tgtNames = [app.targetIDNameMap[id] for id in top_tgtIDs]
    topResults = []

    for idx in range(nbest):
        print('top%d:  %s , %f ,  %s ' % (idx + 1, top_tgtIDs[idx], top_confs[idx], top_tgtNames[idx]))
        entry={}
        entry['ListingId'] = top_tgtIDs[idx]
        entry['ListingTitle'] = top_tgtNames[idx]
        entry['rankingScore'] = float(top_confs[idx])
        topResults.append(entry)
    return jsonify( { 'SearchQuery':keywords, 'SearchRankingResults':topResults} )



@app.route('/api/qna', methods=['GET'])
def questionAnswering():
    #parse QnA task's GET request parameters: e.g.:  /api/qna?question=how does secure pay work&?nbest=5
    keywords = request.args.get('question')
    if 'nbest' in request.args:
      nbest = int(request.args.get('nbest'))
    else:
      nbest = 5

    # inference tensorflow model
    # Get token-ids for the input sentence.
    source_tokens = app.encoder.encode(tf.compat.as_str(keywords).lower())
    src_len = len(source_tokens)
    if src_len > int(app.modelConfigs['max_seq_length']):
      source_tokens = source_tokens[:int(app.modelConfigs['max_seq_length'])]
    else:
      source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * ( int(app.modelConfigs['max_seq_length']) - src_len - 1)
    dict = app.model.get_source_encoding_feed_dict(np.array([source_tokens]), np.array([src_len+1]))
    sourceEncodings = app.sess.run([app.model.src_seq_embedding], feed_dict=dict)
    #sourceEncodings = app.sess.run([app.model.norm_src_seq_embedding], feed_dict=dict)
    sourceEncodings = np.vstack(sourceEncodings)
    distances = np.dot(sourceEncodings, app.targetEncodings.T)
    rankedScore, rankedIdx = data_utils.getSortedResults(distances)
    top_confs = rankedScore[0][:nbest]
    top_tgtIDs = [app.targetIDs[lbl] for lbl in rankedIdx[0][:nbest]]
    top_tgtNames = [app.targetIDNameMap[id] for id in top_tgtIDs]
    topResults = []


    for idx in range(nbest):
        print('top%d:  %s , %f ,  %s ' % (idx + 1, top_tgtIDs[idx], top_confs[idx], top_tgtNames[idx]))
        entry={}
        entry['answerDocId'] = top_tgtIDs[idx]
        entry['answerContent'] = top_tgtNames[idx]
        entry['confidenceScore'] = float(top_confs[idx])
        topResults.append(entry)
    return jsonify( { 'Question':keywords, 'Answers':topResults} )


@app.route('/api/crosslingual', methods=['GET'])
def crosslingualSearch():
    #parse out cross-lingual search task GET request parameters: e.g.:  /api/crosslingual?query=nike运动鞋&?nbest=10
    keywords = request.args.get('query')
    if 'nbest' in request.args:
      nbest = int(request.args.get('nbest'))
    else:
      nbest = 10

    # inference tensorflow model
    # Get token-ids for the input sentence.
    source_tokens = app.encoder.encode(tf.compat.as_str(keywords).lower())
    src_len = len(source_tokens)
    if src_len > int(app.modelConfigs['max_seq_length']):
      source_tokens = source_tokens[:int(app.modelConfigs['max_seq_length'])]
    else:
      source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * ( int(app.modelConfigs['max_seq_length']) - src_len - 1)
    dict = app.model.get_source_encoding_feed_dict(np.array([source_tokens]), np.array([src_len+1]))
    sourceEncodings = app.sess.run([app.model.src_seq_embedding], feed_dict=dict)
    #sourceEncodings = app.sess.run([app.model.norm_src_seq_embedding], feed_dict=dict)
    sourceEncodings = np.vstack(sourceEncodings)
    distances = np.dot(sourceEncodings, app.targetEncodings.T)
    rankedScore, rankedIdx = data_utils.getSortedResults(distances)
    top_confs = rankedScore[0][:nbest]
    top_tgtIDs = [app.targetIDs[lbl] for lbl in rankedIdx[0][:nbest]]
    top_tgtNames = [app.targetIDNameMap[id] for id in top_tgtIDs]
    topResults = []


    for idx in range(nbest):
        print('top%d:  %s , %f ,  %s ' % (idx + 1, top_tgtIDs[idx], top_confs[idx], top_tgtNames[idx]))
        entry={}
        entry['documentId'] = top_tgtIDs[idx]
        entry['documentTitle'] = top_tgtNames[idx]
        entry['confScore'] = float(top_confs[idx])
        topResults.append(entry)
    return jsonify( { 'CrossLingualQuery':keywords, 'SearchResults':topResults} )




@app.route('/', methods=['GET'])
def default():
    return 'Sequence Semantic Embedding NLP toolkit demo webserver. \n For classification task, send GET request with URL of  /api/classify?keywords=hello kitty sunglasses \n For search relevance ranking task, send GET request with  /api/search?query=red nike shoes&?nbest=10 \n For question answering task, send GET request with  /api/qna?question=how does secure pay work&?nbest=5  \n For cross-lingual search task, send  GET request with /api/crosslingual?query=nike运动鞋&?nbest=10 \n'

