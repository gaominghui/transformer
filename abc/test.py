# coding:utf-8
# !/usr/bin/python
# -*- coding:UTF-8 -*-
import operator
from math import radians ,asin,sqrt,sin,cos
import xlrd
import numpy as np
from operator import itemgetter, attrgetter
import time
import datetime

import geohash
import time
import datetime
import pytz
import gensim
import pysnooper
import numpy as np
import tensorflow as tf

def label_smoothing(inputs, epsilon=0.1):
    V = inputs.get_shape().as_list()[-1] # number of channels
    print V
    print ((1-epsilon) * inputs)
    return ((1-epsilon) * inputs) + (epsilon / V)

inputs = tf.convert_to_tensor([[0, 0, 1],
                                [1, 0, 0]], tf.float32)




inputs2 = tf.convert_to_tensor([[0],
                                [0]], tf.float32)






Truth = np.array([[0,0,1,0],
                  [1,0,0,0]])
Pred_logits = np.array([[3.5,2.1,7.89,4.4],
                        [8.00,2.00,6.52,3.14]])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=Truth,logits=Pred_logits)
loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Truth,logits=Pred_logits)
#loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Truth),logits=Pred_logits)

inputs4 = tf.concat((inputs2, inputs), 1)
gamma = tf.get_variable("gamma", 512, initializer=tf.ones_initializer())
aa = tf.random_uniform((), 0, 18, tf.int32)




#inputs=np.array([[[1.0,0.0,0,0],[0.0,1,1,0],[1.0,1,1,1]],[[1.0,0,0,0],[0.0,1,1,0],[1.0,1,1,1]]])
keys=np.array([[[3.0,2,3,3],[4.0,5,6,4],[2.0,2,2,5]],[[3.0,2,3,3],[4.0,5,6,4],[2.0,2,2,5]]])
inputs=np.array([[[3.0,2,3,3],[4.0,5,6,4],[2.0,2,2,5]],[[1.0,0,0,0],[0.0,1,1,0],[1.0,1,1,1]]])
padding_num = -2 ** 32 + 1
with tf.Session() as sess:
    print (sess.run(tf.shape(inputs)))
    sess.run(tf.global_variables_initializer())

    diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
    print (sess.run(diag_vals))
    print (sess.run(tf.shape(diag_vals)))
    print '-----------'
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
    print (sess.run(tril))

    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
    print (sess.run(masks))
    paddings = tf.ones_like(masks) * padding_num
    print (sess.run(paddings))
    print (sess.run(tf.shape(inputs)))
    print(sess.run(tf.equal(masks, 0)))
    print(sess.run(tf.where(tf.equal(masks, 0), paddings, inputs)))

    print '*'*100
    tril = tf.linalg.LinearOperatorLowerTriangular(inputs).to_dense()  # (T_q, T_k)
    print (sess.run(tril))




