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
  train_path = os.path.join(processedDir, "Train")
  dev_path = os.path.join(processedDir, "Eval")
  if not (gfile.Exists(train_path +".target") and gfile.Exists(train_path +".source")):
    corpus_file = os.path.join(rawDir, "DataSet.tar.gz")
    if not gfile.Exists(corpus_file):
      print('Error! No corups file found at: %s' % corpus_file )
      exit(1)
    print("Extracting tar file %s" % corpus_file)
    #extract out the TrainPairs file
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(processedDir)
    #produce the train corpus file
    with codecs.open( train_path + '.source.Corpus', 'w', 'utf-8' ) as srcFile, \
         codecs.open(train_path + '.target.Corpus', 'w', 'utf-8') as tgtFile:
      for line in codecs.open( os.path.join(processedDir, 'TrainPairs'), 'r', 'utf-8'):
        info = line.lower().strip().split('\t') # srcSeq, tgtSeq, tgtId = line.strip().split('\t')
        if len(info) < 2:
          print('Error train pair data:%s' % line)
          continue
        srcFile.write(info[0] + '\n')
        tgtFile.write(info[1] + '\n')
    #produce the eval corpus file
    with codecs.open(dev_path + '.source.Corpus', 'w', 'utf-8') as srcFile, \
         codecs.open(dev_path + '.target.Corpus', 'w', 'utf-8') as tgtFile:
      for line in codecs.open(os.path.join(processedDir, 'EvalPairs'), 'r', 'utf-8'):
        info = line.lower().strip().split('\t') # srcSeq, tgtSeq, tgtId = line.strip().split('\t')
        if len(info) < 2:
          print('Error train pair data:%s' % line)
          continue
        srcFile.write(info[0] + '\n')
        tgtFile.write(info[1] + '\n')

  return train_path, dev_path



def gen_classification_corpus( pairfilename, encodedTargetSpace, encoder, max_seq_length ):
  """

  :param pairfilename:
  :param encoder:
  :param max_seq_length:
  :return:
  """
  Corpus = []
  counter = 0
  for line in codecs.open( pairfilename , "r", 'utf-8'):
    info = line.strip().split('\t')
    if len(info) != 3:
      print("File %s has Bad line of training data:\n %s" % ( pairfilename, line ) )
      continue
    srcSeq, tgtSeq, tgtId = info
    counter += 1
    if counter % 100000 == 0:
      print("  reading data line %d" % counter)
      sys.stdout.flush()
    # verify target sequence correctness
    if tgtId not in set(encodedTargetSpace.keys()):
      print('Error Detected!! trouble in finding targetID in target Space file!! %s' % line)
      continue
    source_tokens = encoder.encode(srcSeq.lower())
    seqlen = len(source_tokens)
    if seqlen > max_seq_length - 1:
      print(
        'Error Deteced!!! \n Source Seq:\n %s \n Its seq length is:%d,  which is longer than MAX_SEQ_LENTH of %d. Try to increase limit!!!!' % (
        srcSeq, seqlen, max_seq_length))
      continue
    source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * (max_seq_length - seqlen - 1)
    Corpus.append( (source_tokens, tgtId ) )
  return Corpus


def get_classification_corpus(processed_data_dir, encoder, max_seq_length):
  """

  :param processed_data_dir: contains TrainPairs, EvalPairs and targetIDs files
  :param encoder:
  :param max_seq_length:
  :return:
  """
  #create Encoded TargetSpace Data
  print("Generating classification training corpus .... ")
  encodedFullTargetSpace = {}
  tgtIdNameMap = {}
  encodedFullTargetFile = codecs.open( os.path.join(processed_data_dir, "encoded.FullTargetSpace"), 'w', 'utf-8')
  for line in codecs.open( os.path.join(processed_data_dir, "targetIDs"), 'r', 'utf-8'):
    tgtSeq, id = line.strip().split('\t')
    token_ids = encoder.encode(tgtSeq.lower())
    seqlen = len(token_ids)
    if seqlen > max_seq_length-1:
      print( 'Error Deteced!!! \n Target:\n %s \n Its seq length is:%d,  which is longer than MAX_SEQ_LENTH of %d. Try to increase limit!!!!' % (tgtSeq, seqlen, max_seq_length ))
      continue
    token_ids =  token_ids + [ text_encoder.EOS_ID ] + [ text_encoder.PAD_ID] * (max_seq_length - seqlen -1)
    encodedFullTargetSpace[id] = token_ids
    tgtIdNameMap[id] = tgtSeq
    decoded_tgt = encoder.decode(token_ids)
    subtoken_strings = [encoder._all_subtoken_strings[i] for i in token_ids]
    #debugging
    # encodedFullTargetFile.write(id + '\t' + tgtSeq.strip() + '\t' + ','.join([str(i) for i in token_ids]) + '\t' + ','.join(subtoken_strings)  + '\t' + decoded_tgt + '\n'  )
    encodedFullTargetFile.write(id + '\t' + tgtSeq.strip() + '\t' + ','.join([str(i) for i in token_ids]) + '\n'  )

  encodedFullTargetFile.close()

  #create Encoded Training Corpus: (srcTokens, srcLen, tgtTokens, tgtLen, RelevanceLabel)
  trainingCorpus = gen_classification_corpus( os.path.join(processed_data_dir, "TrainPairs" ), encodedFullTargetSpace, encoder, max_seq_length)
  #creat evaluation corpus: (srcTokens, srcLen, tgtLabels)
  evalCorpus = []
  for line in codecs.open( os.path.join(processed_data_dir, "EvalPairs" ), "r", 'utf-8'):
    info = line.strip().split('\t')
    if len(info) != 3:
      print("EvalFile has Bad line of training data:\n %s" % ( line ) )
      continue
    srcSeq, tgtSeq, tgtId = info
    # verify target sequence correctness
    if tgtId not in set(encodedFullTargetSpace.keys()):
      print('Error Detected!! trouble in finding evalPairs targetID in target Space file!! %s' % line)
      continue
    source_tokens = encoder.encode(srcSeq.lower())
    seqlen = len(source_tokens)
    if seqlen > max_seq_length - 1:
      print(
        'Error Deteced!!! \n Source Seq:\n %s \n Its seq length is:%d,  which is longer than MAX_SEQ_LENTH of %d. Try to increase limit!!!!' % (
        srcSeq, seqlen, max_seq_length))
      continue
    source_tokens = source_tokens + [text_encoder.EOS_ID] + [text_encoder.PAD_ID] * (max_seq_length - seqlen - 1)
    # get positive sample
    evalCorpus.append((source_tokens, source_tokens.index(text_encoder.PAD_ID) +1, tgtId ) )

  #debugging purpose
  print("evalCorpus[1] is:\n source_tokens: %s \n source_length: %s \n tgtId: %s" % ( str(evalCorpus[1][0]), str(evalCorpus[1][1]), str(evalCorpus[1][2])  ) )

  return trainingCorpus, evalCorpus, encodedFullTargetSpace, tgtIdNameMap


def get_search_corpus(processed_data_dir, encoder, max_seq_length):
  """

  :param processed_data_dir:
  :param encoder:
  :param negative_samples:
  :param max_seq_length:
  :return:
  """
  raise NotImplementedError('Search Ranking Task will be supported very soon.')


def get_questionAnswer_corpus(processed_data_dir, encoder, max_seq_length):
  """

  :param processed_data_dir: contains TrainPairs, EvalPairs and targetIDs files
  :param encoder:
  :param max_seq_length:
  :return:
  """
  # note QnA task's data format is the same as Classification task. So we can reuse it.

  return get_classification_corpus(processed_data_dir, encoder, max_seq_length)



def prepare_raw_data(raw_data_dir, processed_data_dir, vocabulary_size, task_type,  max_seq_length):
  """Get SSE training-Evaluation data into data_dir, create vocabularies and tokenized data.

  Args:
    raw_data_dir:  directory contains the raw zipped dataset.
    processed_data_dir: directory in which the processed data sets will be stored.
    vocabulary_size: size of the vocabulary to create and use if no vocabulary file found in rawdata. Otherwise, use supplied vocabulary file.
    task_type: different task_type has slightly different rawdata format, and need different treatment
               for classification task, usually has TrainPairs, EvalPairs, targetSpaceID file
               for search task,
               for cross-lingual search tasks,
               for question answer tasks,
    max_seq_length: max number of tokens  of a single source/target sequence
  Returns:
    A tuple of 5 elements:
      (1) path to encoded TrainPairs: targetID, Sequence of source token IDs
      (2) path to encoded EvalPairs: targetID, Sequence of source token IDs
      (3) path to encoded full TargetSpaces: targetID, Sequence of target token IDs
      (4) path to the source vocabulary file,
      (5) path to the target vocabulary file.
  """
  # extract corpus to the specified processed directory.
  get_data_set(raw_data_dir, processed_data_dir)

  # generate vocab file if not available, otherwise, use supplied vocab file for encoder
  vocabFile = processed_data_dir + '/vocabulary.txt'
  if  gfile.Exists( vocabFile ):
    print("Loading supplied vocabluary file: %s" % vocabFile )
    encoder = text_encoder.SubwordTextEncoder(filename=vocabFile)
    print("Total vocab size is: %d" % encoder.vocab_size )
  else:
    print("No supplied vocabulary file found. Build new vocabulary based on training data ....")
    token_counts = tokenizer.corpus_token_counts( processed_data_dir + '/*.Corpus', 1000000, split_on_newlines=True)
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size( vocabulary_size, token_counts, 2, 1000 )
    encoder.store_to_file(vocabFile)
    print("New vocabulary constructed.")

  # create training corpus and evaluation corpus per task_type
  if task_type.lower().strip() == "classification":
    train_corpus, dev_corpus, encodedTgtSpace, tgtIdNameMap = get_classification_corpus( processed_data_dir, encoder, max_seq_length)
  elif task_type.lower().strip() in ["ranking", "crosslingual" ]:
    train_corpus, dev_corpus, encodedTgtSpace, tgtIdNameMap = get_search_corpus( processed_data_dir, encoder,  max_seq_length)
  elif task_type.lower().strip()  == "qna":
    train_corpus, dev_corpus, encodedTgtSpace, tgtIdNameMap = get_questionAnswer_corpus(processed_data_dir, encoder, max_seq_length)
  else:
    raise ValueError("Unsupported task_type. Please use one of: classification, search, crosslanguages, questionanswer")

  return encoder, train_corpus, dev_corpus, encodedTgtSpace, tgtIdNameMap




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
  learning_rate,  learning_rate_decay_factor, targetSpaceSize, network_mode, TOP_N = configs
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
  outfile.write( 'network_mode=' + str(network_mode) + '\n')
  outfile.write( 'TOP_N=' + str(TOP_N) + '\n')
  outfile.close()
  return


def load_model_configs(processed_data_dir):
  modelConfig={}
  for line in open(processed_data_dir + '/modelConfig.param', 'rt'):
    if '=' not in line.strip():
      continue
    key, value = line.strip().split('=')
    modelConfig[key]=value
  return modelConfig

def getSortedResults(scores):
  rankedIdx = np.argsort( -scores )
  sortedScore = -np.sort( -scores, axis=1 )
  print('Sample top5 scores:' , sortedScore[0:5])
  return sortedScore, rankedIdx

def computeTopK_accuracy( topk, labels, results ):
  k = min(topk, results.shape[1])
  nbrCorrect=0.0
  for i in xrange(results.shape[0]):
    if labels[i] in results[i][:k]:
      nbrCorrect += 1.0
  return nbrCorrect / float(results.shape[0])

