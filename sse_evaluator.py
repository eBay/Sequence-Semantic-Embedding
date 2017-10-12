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


class Evaluator(object):
  """
    Helper class to evaluate models using an evaluation set.

    Calculate top-n accuracy on the evaluation set.

    evaluator = model.Evaluator(model, eval_src, srcLens, tgtIDs, tgtInputs, tgtLens, tgtID_FullLabelMap , sess, batch_size=2048)

  """
  def __init__(self, model, srcSeqs, srcLens, tgtIDs, target_inputs, target_lens, tgtID_FullLableMap, session, batch_size=2048 ):
    """
    Initializes an Evaluator.

    :param model: save SSE models
    :param srcSeqs: evaluation set source inpput tokens seq
    :param srcLens: length of src sequences
    :param tgtIDs:  labeld tgtIDs for each source inputs
    :param target_inputs, target_lens: full target Space inputs sequences and their lengths
    :param session: current tensorflow session
    :param batch_size: batch_size to run

    """

    self.model = model
    self.tgtIDs = tgtIDs
    self.tgtID_FullLableMap = tgtID_FullLableMap

    #split evaluation set into batches
    srcSeq_batches = [ srcSeqs[k:k + batch_size] for k in range(0, len(srcSeqs), batch_size) ]

    srcLens_batches = [ srcLens[k:k + batch_size] for k in range(0, len(srcSeqs), batch_size) ]

    tgtInput_batches = [ target_inputs for k in range(0, len(srcSeqs), batch_size)]

    tgtLen_batches = [ target_lens for k in range(0, len(srcSeqs), batch_size)]

    self.feed_dicts = list(map(model.get_predict_feed_dict, srcSeq_batches, tgtInput_batches, srcLens_batches, tgtLen_batches))

    self.session = session


  def eval(self, top_n=(1, 3, 10)):
    """
    Obtains predictions for eval set target sequences and compares them to the
    respective previous labels.
    Returns an array of top-n accuracies.
    """
    acc = []
    self.model.set_forward_only(True)
    pred_labels = [ self.session.run( self.model.predicted_labels, feed_dict=d)
                                for d in self.feed_dicts ]

    pred_labels = np.vstack(pred_labels)


    for n in top_n:
      k = 0
      for j in range(len(self.tgtIDs)):
        if self.tgtID_FullLableMap[self.tgtIDs[j]] in pred_labels[j][:n]:
          k += 1
      acc.append(float(k) / len(self.tgtIDs))
    return acc
