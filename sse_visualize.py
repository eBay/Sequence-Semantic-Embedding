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
# @Date:   2017-11-08
#
##################################################################################
import sys
import codecs
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif']=['SimHei']

TOPN=8000

def visualize(sseEncodingFile, gneratedImageFile):
    print("Loading embingfile: %s" % sseEncodingFile)
    raw_seq, sse = load_embeddings(sseEncodingFile)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    print("fitting tsne...")
    Y = tsne.fit_transform(sse)
    print("plotting...")
    ChineseFont = FontProperties('SimHei')
    plt.title("SSE Representations", fontdict={'fontsize': 16})
    plt.figure(figsize=(100, 100))  # in inches
    for label, x, y in zip(raw_seq, Y[:, 0], Y[:, 1]):
        plt.scatter(x,y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points',
                     fontproperties=ChineseFont,
                     ha='right',
                     va='bottom')
    plt.savefig(gneratedImageFile, format='png')
    plt.show()


def load_embeddings(file_name):
    raw_seq, sse = [], []
    for line in codecs.open(file_name, 'r', 'utf-8').readlines():
        info = line.strip().split('\t')
        if len(info) !=3:
            print("Error line with len:%d in SSE encoding file: %s" % (len(info), line) )
            continue
        tgtid, seq, embedding = info
        raw_seq.append(seq)
        sse.append( [ float(x) for x in embedding.split(',') ] )
    xSSE = np.asarray(sse).astype('float64')
    return raw_seq[-TOPN:], xSSE[-TOPN:]



if __name__ == "__main__":
  visualize( sys.argv[1] , sys.argv[2] )

