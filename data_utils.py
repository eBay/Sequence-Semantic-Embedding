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


"""Utilities for extracting and preprocessing training and evaluation data with tokenizing, encoding inputs etc.
 The DataSet.tar.gz file in the rawdata folder contains a sample dataset used in classification task.  It contains
 three files: TrainPairs, EvalPairs and targetID.
    * The TrainPairs, EvalPairs are training/evaluation corpus data in the
  format of tsv file with columns of SourceSequence, TargetSequence, TargetSeqId.
    * The targetID file contains the whole target sequence space and their IDs in the format of: targetSequence, targetSequenceID.

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from builtins import str
from builtins import str
import gzip
import os
import re
import tarfile
import codecs
import random
import numpy as np
import sys
import text_encoder
import tokenizer
import time

from six.moves import urllib

from tensorflow.python.platform import gfile


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_data_set(rawDir, processedDir):
  if not (gfile.Exists(processedDir + "/TrainPairs") and gfile.Exists(processedDir + "/vocabulary.txt")):
    corpus_file = os.path.join(rawDir, "DataSet.tar.gz")
    if not gfile.Exists(corpus_file):
      print('Error! No corups file found at: %s' % corpus_file )
      exit(1)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(processedDir)
    with codecs.open( processedDir + '/sourceSeq.Corpus', 'w', 'utf-8' ) as srcFile, \
         codecs.open(processedDir + '/targetSeq.Corpus', 'w', 'utf-8') as tgtFile:
      # produce source seq corpus
      for line in codecs.open( os.path.join(processedDir, 'TrainPairs'), 'r', 'utf-8'):
        info = line.strip().split('\t') # srcSeq,  tgtId = line.strip().split('\t')
        if len(info) < 2:
          print('Error train pair data:%s' % line)
          continue
        srcFile.write(info[0].lower() + '\n')
      for line in codecs.open(os.path.join(processedDir, 'EvalPairs'), 'r', 'utf-8'):
        info = line.strip().split('\t')
        if len(info) < 2:
          continue
        srcFile.write(info[0].lower() + '\n')
      # produce the target corpus file
      for line in codecs.open(os.path.join(processedDir, 'targetIDs'), 'r', 'utf-8'):
        info = line.strip().split('\t') # tgtSeq, tgtId = line.strip().split('\t')
        if len(info) < 2:
          print('Error in targetIDs file:%s' % line)
          continue
        tgtFile.write(info[0].lower() + '\n')
  return


def gen_postive_corpus( pairfilename, encodedTargetSpace, encoder, max_seq_length ):
  """

  :param pairfilename:
  :param encoder:
  :param max_seq_length:
  :return:
  """
  Corpus = []
  counter = 0
  tgtIdSets = set(encodedTargetSpace.keys())
  for line in codecs.open( pairfilename , "r", 'utf-8'):
    info = line.strip().split('\t')
    if len(info) != 2:
      print("File %s has Bad line of training data:\n %s" % ( pairfilename, line ) )
      continue
    srcSeq,  tgtIds = info
    counter += 1
    if counter % 100000 == 0:
      print("  reading data line %d" % counter)
      sys.stdout.flush()
    # verify target sequence correctness
    verifiedTgtIds = []
    for tgtid in tgtIds.split('|'):
      if tgtid not in tgtIdSets:
        print('Error Detected!! trouble in finding targetID in target Space file!! %s' % line)
        continue
      else:
        verifiedTgtIds.append(tgtid)
    if len(verifiedTgtIds) == 0:
      print('Not found any verified tgtIDs in line:%s' % line)
      continue
    source_tokens = encoder.encode(srcSeq.lower())
    seqlen = len(source_tokens)
    if seqlen > max_seq_length - 1:
      print(
        'Error Deteced!!! \n Source Seq:\n %s \n Its seq length is:%d,  which is longer than MAX_SEQ_LENTH of %d. Try to increase limit!!!!' % (
        srcSeq, seqlen, max_seq_length))
      continue
    source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * (max_seq_length - seqlen - 1)
    Corpus.append( (source_tokens, verifiedTgtIds ) )
  return Corpus


def prepare_raw_data(raw_data_dir, processed_data_dir, vocabulary_size, neg_samples, max_seq_length):
  """
  Get SSE training, and Evaluation related data, create tokenizer and vocabulary.

  :param raw_data_dir:
  :param processed_data_dir:
  :param vocabulary_size:
  :param neg_samples:
  :param max_seq_length:
  :return:
  """
  # unzip corpus to the specified processed directory.
  get_data_set(raw_data_dir, processed_data_dir)

  # generate vocab file if not available, otherwise, use supplied vocab file for encoder
  vocabFile = processed_data_dir + '/vocabulary.txt'
  if gfile.Exists(vocabFile):
    print("Loading supplied vocabluary file: %s" % vocabFile)
    encoder = text_encoder.SubwordTextEncoder(filename=vocabFile)
    print("Total vocab size is: %d" % encoder.vocab_size)
  else:
    print("No supplied vocabulary file found. Build new vocabulary based on training data ....")
    token_counts = tokenizer.corpus_token_counts(processed_data_dir + '/*.Corpus', 1000000, split_on_newlines=True)
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(vocabulary_size, token_counts, 2, 1000)
    encoder.store_to_file(vocabFile)
    print("New vocabulary constructed.")

  # create encoded TargetSpace Data
  encodedFullTargetSpace = {}
  tgtIdNameMap = {}
  encodedFullTargetFile = codecs.open(os.path.join(processed_data_dir, "encoded.FullTargetSpace"), 'w', 'utf-8')
  for line in codecs.open(os.path.join(processed_data_dir, "targetIDs"), 'r', 'utf-8'):
    tgtSeq, id = line.strip().split('\t')
    token_ids = encoder.encode(tgtSeq.lower())
    seqlen = len(token_ids)
    if seqlen > max_seq_length - 1:
      print(
        'Error Detected!!! \n Target:\n %s \n Its seq length is:%d,  which is longer than MAX_SEQ_LENTH of %d. Try to increase limit!!!!' % (
        tgtSeq, seqlen, max_seq_length))
      continue
    token_ids = token_ids + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * (max_seq_length - seqlen - 1)
    encodedFullTargetSpace[id] = token_ids
    tgtIdNameMap[id] = tgtSeq
    encodedFullTargetFile.write(id + '\t' + tgtSeq.strip() + '\t' + ','.join([str(i) for i in token_ids]) + '\n')
  encodedFullTargetFile.close()

  # creat positive Evaluation corpus: (source_tokens, verifiedTgtIds )
  evalCorpus = gen_postive_corpus(os.path.join(processed_data_dir, "EvalPairs"), encodedFullTargetSpace, encoder,
                                  max_seq_length)

  # create positive Training Corpus: (source_tokens, verifiedTgtIds )
  trainCorpus = gen_postive_corpus(os.path.join(processed_data_dir, "TrainPairs"), encodedFullTargetSpace,
                                         encoder, max_seq_length)
  return encoder, trainCorpus, evalCorpus, encodedFullTargetSpace, tgtIdNameMap



def load_encodedTargetSpace(processed_data_dir):
  """

  :param processed_data_dir:
  :return:
  """
  vocabFile = processed_data_dir + '/vocabulary.txt'
  if  gfile.Exists( vocabFile ):
    encoder = text_encoder.SubwordTextEncoder(filename=vocabFile)
    print("Loaded  vocab size is: %d" % encoder.vocab_size )
  else:
    raise  ValueError("Error!! Could not found vaculary file in model folder.")
  encodedTgtSpace = {}
  tgtID_Name_Map = {}
  tgtEncodeFile = os.path.join(processed_data_dir, "encoded.FullTargetSpace")
  if not gfile.Exists(tgtEncodeFile):
    raise ValueError("Error! could not found encoded.FullTargetSpace in model folder.")
  print("Loading full target space index ...")
  for line in codecs.open( tgtEncodeFile, 'r',  'utf-8'):
    tgtId, tgtName, tgtEncoding = line.strip().split('\t')
    tgtID_Name_Map[tgtId] = tgtName
    encodedTgtSpace[tgtId] = [ int(i) for i in tgtEncoding.split(',') ]
  return encoder, encodedTgtSpace, tgtID_Name_Map



def save_model_configs(processed_data_dir, configs):
  max_seq_length, max_gradient_norm, vocabsize, embedding_size, \
  encoding_size, src_cell_size,  tgt_cell_size, num_layers, \
  learning_rate,  learning_rate_decay_factor, targetSpaceSize, network_mode, TOP_N, alpha, neg_samples = configs
  outfile = codecs.open( os.path.join(processed_data_dir,'modelConfig.param'), 'w', 'utf-8')
  outfile.write( 'max_seq_length=' + str(max_seq_length) + '\n')
  outfile.write( 'max_gradient_norm=' + str(max_gradient_norm) + '\n')
  outfile.write( 'vocabsize=' + str(vocabsize) + '\n')
  outfile.write( 'embedding_size=' + str(embedding_size) + '\n')
  outfile.write( 'encoding_size=' + str(encoding_size) + '\n')
  outfile.write( 'src_cell_size=' + str(src_cell_size) + '\n')
  outfile.write( 'tgt_cell_size=' + str(tgt_cell_size) + '\n')
  outfile.write( 'num_layers=' + str(num_layers) + '\n')
  outfile.write( 'learning_rate=' + str(learning_rate) + '\n')
  outfile.write( 'learning_rate_decay_factor=' + str(learning_rate_decay_factor) + '\n')
  outfile.write( 'targetSpaceSize=' + str(targetSpaceSize) + '\n')
  outfile.write( 'network_mode=' + str(network_mode) + '\n' )
  outfile.write( 'TOP_N=' + str(TOP_N) + '\n' )
  outfile.write( 'alpha=' + str(alpha) + '\n' )
  outfile.write( 'neg_samples=' + str(neg_samples) + '\n' )
  outfile.close()
  return


def load_model_configs(processed_data_dir):
  modelConfig={}
  for line in codecs.open(os.path.join(processed_data_dir, 'modelConfig.param'), 'r', 'utf-8').readlines():
    if '=' not in line.strip():
      continue
    key, value = line.strip().split('=')
    modelConfig[key]=value
  return modelConfig


def getSortedResults(scores):
  rankedIdx = np.argsort( -scores )
  sortedScore = -np.sort( -scores, axis=1 )
  #print('Sample top5 scores:' , sortedScore[0:5])
  return sortedScore, rankedIdx


def computeTopK_TightVersion_accuracy( topk, labels, results ):
  """
  :param topk:
  :param labels: two demensions. Each entry can have multiple correct labels
  :param results:
  :return:
  """
  assert len(labels) == len(results)
  k = min(topk, results.shape[1])
  totalCorrect=0.0
  for i in range(results.shape[0]):
    curCorrect = 0.0
    for correctLabel in labels[i]:
      if correctLabel in results[i][:k]:
        curCorrect += 1.0
    totalCorrect += curCorrect / len(labels[i])
  return totalCorrect / float(results.shape[0])


def computeTopK_accuracy( topk, labels, results ):
  """
  :param topk:
  :param labels: two demensions. Each entry can have multiple correct labels
  :param results:
  :return:
  """
  assert len(labels) == len(results)
  k = min(topk, results.shape[1])
  totalCorrect=0.0
  for i in range(results.shape[0]):
    for correctLabel in labels[i]:
      if correctLabel in results[i][:k]:
        totalCorrect += 1.0
        break
  return totalCorrect / float(results.shape[0])
