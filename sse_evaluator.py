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



"""
 Accuracy evaluator for Sequence Semantic Embedding model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import map
from builtins import range
from builtins import object
import codecs

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import sse_model
import text_encoder
import math

class Evaluator(object):
  """
    Helper class to evaluate models using an evaluation set.

    Calculate top-n accuracy on the evaluation set.

    evaluator = model.Evaluator(model, eval_src, srcLens, tgtIDs, tgtInputs, tgtLens, tgtID_FullLabelMap , sess, batch_size=2048)

  """
  def __init__(self, model, eval_corpus, tgtIndexFile,  session  ):
    """
    Initializes an Evaluator.

    :param model: SSE model
    :param eval_corpus: eval corpus contains source_tokens, source_seq_length, correct_targetIds
    :param tgtIndexFile: encoded full targetSpace Index file. Format: targetID, targetSequence, targetEncodings
    :param session:
    :param batch_size:
    """

    self.model = model
    srcSeq_batch = [  entry[0]  for entry in  eval_corpus ]
    srcLens_batch = [ entry.index(text_encoder.PAD_ID) +1   for entry in  srcSeq_batch ]
    self.feed_dict = model.get_source_encoding_feed_dict(srcSeq_batch, srcLens_batch)
    self.session = session

    self.targetEncodings = []
    self.targetIDs = []
    self.idLabelMap = {}
    idx=0
    for line in codecs.open(tgtIndexFile, 'r', 'utf-8').readlines():
      info = line.strip().split('\t')
      if len(info) != 3:
        print('Error in targetIndexFile! %s' % line)
        continue
      tgtid, tgtseq, tgtEncoding = info[0], info[1], info[2]
      self.targetIDs.append(tgtid)
      self.targetEncodings.append( [ float(f) for f in tgtEncoding.strip().split(',') ] )
      self.idLabelMap[tgtid] = idx
      idx += 1
    self.eval_Labels = [ [ self.idLabelMap[tgtid] for tgtid in entry[1] ] for entry in eval_corpus ]

    self.targetEncodings = np.array( self.targetEncodings )


  def eval(self, top_n=(1, 3, 10)):
    """
    Obtains predictions for eval set target sequences and compares them to the
    respective previous labels.
    Returns an array of top-n accuracies.
    """
    self.model.set_forward_only(True)
    sourceEncodings = self.session.run( [self.model.src_seq_embedding], feed_dict= self.feed_dict )
    #sourceEncodings = self.session.run( [self.model.norm_src_seq_embedding], feed_dict= self.feed_dict )
    sourceEncodings = np.vstack( sourceEncodings )
    batchSize = 600
    top1Acc, top3Acc, top10Acc = [], [], []
    for batchId in range(math.ceil( len(sourceEncodings) / batchSize )):
      batchSourceEncodings = sourceEncodings[batchId * batchSize: (batchId +1) * batchSize]
      distances = np.dot( batchSourceEncodings, self.targetEncodings.T)
      rankedScore, rankedIdx = data_utils.getSortedResults(distances)
      top1Acc.append( data_utils.computeTopK_accuracy(1, self.eval_Labels, rankedIdx))
      top3Acc.append( data_utils.computeTopK_accuracy(3, self.eval_Labels, rankedIdx))
      top10Acc.append( data_utils.computeTopK_accuracy(10, self.eval_Labels, rankedIdx))
    return [ np.mean(top1Acc), np.mean(top3Acc), np.mean(top10Acc) ]
