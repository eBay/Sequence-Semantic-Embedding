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

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_BOS = "_BOS"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# Regular expressions used to tokenize.
def repl(m):
  return m.group(1) + " " + m.group(2) + " " + m.group(3)

def text_normalize(rawstr):
  tnstring = rawstr.lower()
  tnstring = re.sub("[^a-z0-9':#,$-]", " ", tnstring)
  tnstring = re.sub("\\s+", " ", tnstring).strip()
  return tnstring

_WORD_SPLIT = re.compile("([.,!?\"';-@#)(])")
_DIGIT_RE = re.compile(r"\d")


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
    #produce the train dataset file
    with codecs.open( train_path + '.source', 'w', 'utf-8' ) as srcFile, \
         codecs.open(train_path + '.target', 'w', 'utf-8') as tgtFile:
      for line in codecs.open( os.path.join(processedDir, 'TrainPairs'), 'r', 'utf-8'):
        if len(line.strip().split('\t')) != 3:
          print('Error train pair data:%s' % line)
          continue
        srcSeq, tgtSeq, tgtId = line.strip().split('\t')
        srcSeq = text_normalize(srcSeq)
        srcFile.write(srcSeq + '\n')
        tgtFile.write(tgtId + '\n')
    #produce the eval dataset file
    with codecs.open(dev_path + '.source', 'w', 'utf-8') as srcFile, \
         codecs.open(dev_path + '.target', 'w', 'utf-8') as tgtFile:
      for line in codecs.open(os.path.join(processedDir, 'EvalPairs'), 'r', 'utf-8'):
        if len(line.strip().split('\t')) != 3:
          print('Error dev pair data:%s' % line)
          continue
        srcSeq, tgtSeq, tgtId = line.strip().split('\t')
        srcSeq = text_normalize(srcSeq)
        srcFile.write(srcSeq + '\n')
        tgtFile.write(tgtId + '\n')

  return train_path, dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  sentence_normed = text_normalize(sentence)
  #sentence_normed = sentence.lower()
  for space_separated_fragment in sentence_normed.split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        line = line.strip().split('\t')[0]
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
      vocab_list = _START_VOCAB + sorted_vocab
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
        print("Corpus %s has %d tokens, %d uniq words, %d vocab at cutoff %d." % (
        data_path, sum(vocab.values()), len(vocab),  max_vocabulary_size, vocab[sorted_vocab[max_vocabulary_size - len(_START_VOCAB)]] ) )
      else:
        print("Corpus %s has %d tokens, %d uniq words, %d vocab at cutoff %d." % (
        data_path, sum(vocab.values()), len(vocab), len(vocab), 0))

      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"shoes": 0, "shirt": 1}, and this function will
  also return the reversed-vocabulary ["shoes", "shirt"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "want", "buy", "shoes"] and with vocabulary {"I": 1, "want": 2,
  "buy": 4, "shoes": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def encode_data_to_token_ids(raw_data_path, encoded_path, vocabulary_path, targetSet,
                      tokenizer=None, normalize_digits=True):
  """encode raw train/eval pair data file  into token-ids using given vocabulary file.
     also check if labeled targetID is valid contained in targetSet

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to encoded_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    raw_data_path: path to the train/eval pair data file in src, tgt, targetID format.
    target_path: path where the file with targetID, token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    targetSet: valid targetID collection.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(encoded_path):
    print("Tokenizing data in %s" % raw_data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with codecs.open(encoded_path, 'w', 'utf-8') as tokens_file:
      counter = 0
      for line in codecs.open( raw_data_path, 'r', 'utf-8'):
        counter += 1
        if counter % 100000 == 0:
          print("  tokenizing line %d" % counter)
        if len(line.strip().split('\t')) != 3:
          print('Error data:%s' % line)
          continue
        src, tgt, tgtID = line.strip().split('\t')
        if tgtID not in targetSet:
          print('Error!!! %s with %s not found in full targetID file!!' % (tgt, tgtID))
          continue
        else:
          src = text_normalize(src)
          token_ids = sentence_to_token_ids(src, vocab, normalize_digits)
          token_ids = [BOS_ID] + token_ids + [EOS_ID]
          tokens_file.write(tgtID + '\t' + " ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_raw_data(raw_data_dir, processed_data_dir , src_vocabulary_size, tgt_vocabulary_size, tokenizer=None):
  """Get SSE training-Evaluation data into data_dir, create vocabularies and tokenize data.

  Args:
    raw_data_dir:  directory contains the raw zipped dataset.
    processed_data_dir: directory in which the processed data sets will be stored.
    src_vocabulary_size: size of the source sequence vocabulary to create and use.
    tgt_vocabulary_size: size of the target sequence vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

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

  # Create vocabularies of the appropriate sizes.
  tgt_vocab_path = os.path.join(processed_data_dir, "vocab.tgt" )
  src_vocab_path = os.path.join(processed_data_dir, "vocab.src" )
  create_vocabulary(tgt_vocab_path, os.path.join(processed_data_dir, "targetIDs"), tgt_vocabulary_size, tokenizer, normalize_digits=False)
  create_vocabulary(src_vocab_path, os.path.join(processed_data_dir, "Train.source"), src_vocabulary_size, tokenizer, normalize_digits=False)

  #create Encoded TargetSpace file
  encodedFullTargetSpace_path = os.path.join(processed_data_dir, "encoded.FullTargetSpace")
  tgt_vocab, _ = initialize_vocabulary(tgt_vocab_path)
  targetIDs = set()
  with codecs.open( encodedFullTargetSpace_path, 'w', 'utf-8') as tokens_file:
    for line in codecs.open( os.path.join(processed_data_dir, "targetIDs"), 'r', 'utf-8'):
      tgtSeq, id = line.strip().split('\t')
      token_ids = sentence_to_token_ids(tgtSeq, tgt_vocab, normalize_digits=False)
      token_ids = [BOS_ID] + token_ids + [EOS_ID]
      tokens_file.write( id + '\t' + " ".join([str(tok) for tok in token_ids]) + "\n")
      targetIDs.add(id)

  # Create Encoded TrainPairFile
  encoded_train_pair_path = os.path.join(processed_data_dir, "encoded.TrainPairs")
  raw_train_pair_path = os.path.join(processed_data_dir, 'TrainPairs')
  encode_data_to_token_ids(raw_train_pair_path, encoded_train_pair_path, src_vocab_path, targetIDs, normalize_digits=False)

  # Create Encoded EvalPairFile
  encoded_eval_pair_path = os.path.join(processed_data_dir, "encoded.EvalPairs")
  raw_eval_pair_path = os.path.join(processed_data_dir, 'EvalPairs')
  encode_data_to_token_ids(raw_eval_pair_path, encoded_eval_pair_path, src_vocab_path, targetIDs, normalize_digits=False)


  return (encoded_train_pair_path, encoded_eval_pair_path,
          encodedFullTargetSpace_path,
          src_vocab_path, tgt_vocab_path)

