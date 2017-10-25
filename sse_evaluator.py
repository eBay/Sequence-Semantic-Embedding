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
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import sse_model
import text_encoder

class Evaluator(object):
  """
    Helper class to evaluate models using an evaluation set.

    Calculate top-n accuracy on the evaluation set.

    evaluator = model.Evaluator(model, eval_src, srcLens, tgtIDs, tgtInputs, tgtLens, tgtID_FullLabelMap , sess, batch_size=2048)

  """
  def __init__(self, model, eval_corpus, encodedTgtSpace,  session ):
    """
    Initializes an Evaluator.

    :param model: SSE model
    :param eval_corpus: eval corpus contains source_tokens, source_seq_length, correct_targetIds
    :param encodedTgtSpace: Full TargetSpace encodings. A Map indexed by targetIds
    :param session:
    :param batch_size:
    """

    self.model = model
    fullTgtIdList = list(encodedTgtSpace.keys())
    self.tgtID_FullLableMap = { tgtid:idx for (idx,tgtid) in enumerate(fullTgtIdList) }

    #split evaluation set into batches
    srcSeq_batches = [  entry[0]  for entry in  eval_corpus ]
    srcLens_batches = [ entry[1]  for entry in  eval_corpus ]
    self.tgtIDs = [ entry[2] for entry in eval_corpus ]

    tgtInput_batches = [ encodedTgtSpace[tgtid] for tgtid in fullTgtIdList  ]
    tgtLen_batches = [ encodedTgtSpace[tgtid].index(text_encoder.PAD_ID) + 1 for tgtid in fullTgtIdList ]

    self.feed_dict = model.get_predict_feed_dict(srcSeq_batches, tgtInput_batches, srcLens_batches, tgtLen_batches)
    self.session = session

    print("Raw eval_corpus[1] is: %s" % str(eval_corpus[1]) )
    print("srcSeq_batches[1] is: %s" % str(srcSeq_batches[1]) )
    print("srcLen_batches[1] is: %s" % str(srcLens_batches[1]))

    print("full targetId space len is:%d . First 5 keys of targetIds are: %s" % ( len(fullTgtIdList), str(fullTgtIdList[:5])  )  )


  def eval(self, top_n=(1, 3, 10)):
    """
    Obtains predictions for eval set target sequences and compares them to the
    respective previous labels.
    Returns an array of top-n accuracies.
    """
    acc = []
    self.model.set_forward_only(True)
    pred_labels = self.session.run( self.model.predicted_labels, feed_dict= self.feed_dict )
    # pred_labels = np.vstack(pred_labels)

    for n in top_n:
      k = 0
      for j in range(len(self.tgtIDs)):
        if self.tgtID_FullLableMap[self.tgtIDs[j]] in pred_labels[j][:n]:
          k += 1
      acc.append(float(k) / len(self.tgtIDs))
    return acc
