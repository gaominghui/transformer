# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
import logging

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
#获取训练数据,train_batches,num_train_batches训练数据有多少个batch,num_train_samples一共有多少个训练样本
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
#获取评估数据eval_batches,num_eval_batches验证数据有多少个batch,num_eval_samples一共有多少个评估样本
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)



# create a iterator of the correct shape and type
#为构造迭代器
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
#源语言xs,目标语言ys
xs, ys = iter.get_next()
#初始化构造器
train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
#transformer模型
m = Transformer(hp)
#训练
loss, train_op, global_step, train_summaries = m.train(xs, ys)
#评估
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)

    if ckpt is None:
        logging.info("Initializing from scratch")
        #初始化tensorflow参数
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    #初始化训练数据
    sess.run(train_init_op)
    #每个epoch都要训练一遍所有数据
    total_steps = hp.num_epochs * num_train_batches
    #训练
    _gs = sess.run(global_step)
    #tqdm是一个python进度条库
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)
        #每隔num_train_batches次计算一次
        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            #计算loss
            _loss = sess.run(loss) # train loss

            logging.info("# test evaluation")
            #评估
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            #保存_eval_summaries，_gs到指定的文件
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            #num_eval_batches：验证数据一共有多少个batch
            #num_eval_samples:验证数据一共有个多少个样本
            #y_hat:对验证数据的预测
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            #新建评估目录
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)

            translation = os.path.join(hp.evaldir, model_output)
            #将翻译出的结果hypotheses写回
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to translation")
            #计算bleu
            calc_bleu(hp.eval3, translation)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            #重置训练数据的iterator
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
