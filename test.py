# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf

from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
#load 超参数
load_hparams(hp, hp.ckpt)

logging.info("# Prepare test batches")
#测试数据集
test_batches, num_test_batches, num_test_samples  = get_batch(hp.test1, hp.test1,
                                              100000, 100000,
                                              hp.vocab, hp.test_batch_size,
                                              shuffle=False)
#测试数据集爹迭代器
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

#初始化迭代器
test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
#transformer 模型
m = Transformer(hp)
#调用eval函数给测试集预测
y_hat, _ = m.eval(xs, ys)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()
    #通过checkpoint 恢复session,模型,参数
    saver.restore(sess, ckpt)
    #初始化
    sess.run(test_init_op)

    logging.info("# get hypotheses")
    #预测
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.idx2token)

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    #计算结果写入磁盘
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))

    logging.info("# calc bleu score and append it to translation")
    #计算bleu
    calc_bleu(hp.test2, translation)

