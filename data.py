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
# @Date:   2017-07-24
#
##################################################################################



from __future__ import print_function

import os, json
import data_utils
import numpy
import text_encoder
import tokenizer


class Data(object):
	def __init__(self, work_dir, rawdata_dir, rawvocabsize , max_seq_length):
		json_path = work_dir+'/compressed'
		if os.path.exists(json_path):
			# load data from json
			print('loading saved json data from %s' % json_path)
			with open(json_path, 'r') as fin:
				gdict = json.load(fin)
				for name, val in gdict.items():
					setattr(self, name, val)
			# setup encoder from vocabulary file
			vocabFile = work_dir + '/vocabulary.txt'
			if os.path.exists(vocabFile):
				print("Loading supplied vocabluary file: %s" % vocabFile)
				encoder = text_encoder.SubwordTextEncoder(filename=vocabFile)
				print("Total vocab size is: %d" % encoder.vocab_size)
			else:
				print("No supplied vocabulary file found. Build new vocabulary based on training data ....")
				token_counts = tokenizer.corpus_token_counts(work_dir + '/*.Corpus', 2000000, split_on_newlines=True)
				encoder = text_encoder.SubwordTextEncoder.build_to_target_size(rawvocabsize, token_counts, 2, 1000)
				encoder.store_to_file(vocabFile)
				print("New vocabulary constructed.")
			self.encoder =  encoder
			self.max_seq_length = int(self.max_seq_length)
			self.vocab_size = encoder.vocab_size
			print('-')
			print('Vocab size:', self.vocab_size, 'unique words')
			print('-')
			print('Max allowed sequence length:', self.max_seq_length )
			print('-')
		else:
			print('generating data from data path: %s' % rawdata_dir)
			encoder, trainCorpus, evalCorpus, encodedFullTargetSpace, tgtIdNameMap = data_utils.prepare_raw_data(
				rawdata_dir, work_dir, rawvocabsize, max_seq_length)
			self.encoder = encoder
			self.rawTrainPosCorpus = trainCorpus
			self.rawEvalCorpus = evalCorpus
			self.max_seq_length = max_seq_length
			self.encodedFullTargetSpace = encodedFullTargetSpace
			self.tgtIdNameMap = tgtIdNameMap
			self.vocab_size = encoder.vocab_size
			self.fullSetTargetIds = list( encodedFullTargetSpace.keys() )
			self.rawnegSetLen = len(self.fullSetTargetIds)
			print('-')
			print('Vocab size:', self.vocab_size, 'unique words')
			print('-')
			print('Max allowed sequence length:', self.max_seq_length )
			print('-')
			gdict = {}
			for name, attr in self.__dict__.items():
				if not name.startswith("__") and name != 'encoder':
					if not callable(attr) and not type(attr) is staticmethod:
						gdict[name] = attr
			with open(json_path, 'w') as fout:
				json.dump(gdict, fout)
			print('Processed data dumped')


	def get_train_batch(self, batch_size):
		num_samples = len(self.rawTrainPosCorpus)
		idx = numpy.random.randint(0, num_samples-batch_size)+batch_size
		train_batch = self.rawTrainPosCorpus[idx:idx+batch_size]
		source_inputs, tgt_inputs, labels =  [], [], []
		for pos_entry in train_batch:
			source_tokens, verifiedTgtIds = pos_entry
			curPosTgtId = numpy.random.choice(verifiedTgtIds)
			posSets = set(verifiedTgtIds)
			# add current positive pair
			source_inputs.append(source_tokens)
			tgt_inputs.append(self.encodedFullTargetSpace[curPosTgtId])
			labels.append(1.0)
			# add negative pairs as the pair-wise anchor for current positive sample:
			negTgt = self.fullSetTargetIds[numpy.random.randint(0, self.rawnegSetLen)]
			while negTgt in posSets:
				negTgt = self.fullSetTargetIds[numpy.random.randint(0, self.rawnegSetLen)]
			source_inputs.append(source_tokens)
			tgt_inputs.append(self.encodedFullTargetSpace[negTgt])
			labels.append(0.0)
		return source_inputs, tgt_inputs,  labels

	def get_test_batch(self, batch_size):
		num_samples = len(self.rawEvalCorpus)
		idx = numpy.random.randint(0, num_samples-batch_size)+batch_size
		eval_batch = self.rawEvalCorpus[idx:idx+batch_size]
		return eval_batch


	def get_hard_learning_train_batch(self, batch_size):
		print('Not implemented yet.')
		return
	
	def compute_training_confusion_sampes(self):
		print('Not implemented yet.')
		return 