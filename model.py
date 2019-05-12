# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    Transformer中两个关键的指示表示，training和causality
    training
    --------------------
    |     |encode|decode|
     --------------------
    | train| true|true  |
     --------------------
    |  test| false|false|
     --------------------

    training用来指示dropout的模式，为true表示训练模式，false表示推断模式。
     causality
    --------------------
    |     |encode|decode|
     --------------------
    | train| false|true |
     --------------------
    |  test| false|true|
     --------------------
     causalit用来指示mask的模式，当在encode的过程中，可以考虑上下文的信息，
     当在decode的过程中，只可以考虑上文的信息，不能考虑下文的信息。


    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            #x 经过编码后的句子
            # seqlens :句子长度
            #sents1:原始句子
            #对应data_load.py的generator_fn
            x, seqlens, sents1 = xs

            # embedding
            #找到每个词对应的embeding向量,大小是[batch_size,seqlens,d_model]
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            #scale 的意义？
            enc *= self.hp.d_model**0.5 # scale
            #位置编码,maxlen1 = 100 句子的最长长度,位置编码跟句子中词的位置有关，跟词无关
            enc += positional_encoding(enc, self.hp.maxlen1)
            #dropout
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1

    def decode(self, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

            decoder_inputs, y, seqlens, sents2 = ys

            # embedding,同encoder
            #dec的shape : [batch_size,目标语言句子的长度,512]
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            #同 encoder
            dec *= self.hp.d_model ** 0.5  # scale
            #position encoding
            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    #decoder 的  多头 self-attention
                    #dec 的shape : [batch_size,目标语言句子的长度,512]
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    #encoder-decoder attention
                    #helps the decoder focus on appropriate places in the input sequence
                    #encoder-decoder atteion 帮助decoder 部分 找到intpu中对应的部分
                    #dec 的shape : [batch_size,目标语言句子的长度,512]
                    #memory的shape : [batch_szie, 源语言句子的长度, 512]
                    #产出的dec的shape:[batch_size,目标语言句子的长度,512]
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    #dec shape :[batch_size,目标语言句子的长度,512]
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        #dec shape :[batch_size,senten_len,d_model]
        #weghts : [word_size,d_model] ，转置后 [d_model, word_size]
        #logits : [batch_size,senten_len,word_size]
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        #y_hat : [batch-size, senten_len]
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        #encode step
        memory, sents1 = self.encode(xs)
        #decode step
        logits, preds, y, sents2 = self.decode(ys, memory)

        # train scheme
        #y shape :[batch_size,senten_length]
        #label_smoothing 将onehot值变成了连续的float值
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        #y_ [batch_szie,senten_length,vocab_szie]
        #logits : [batch-szie,senten_length,vocab_szie]
        #ce :[batch_size,senten_length]
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        # tf.not_equal 返回逐个元素的布尔值 tf.to_float 将其转化成0,1
        #nopadding :[batch_szie,senten_length]
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        #ce : [batch-szie,senten_length]
        #nopadding :[batch-szie,senten_length]
        #reduce_sum :不添加参数的话，会逐个元素累加获得一个数值。
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        #noam_scheme 会影响学习率，由小变大，再由大变小
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        #docder_inputs shape :[batch_size,senten_length]
        decoder_inputs, y, y_seqlen, sents2 = ys
        #xs[0]  shape :[batch_szie,senten_length]
        #decoder_inputs shape: [batch_size,1]  且均为0
        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        #预测
        for _ in tqdm(range(self.hp.maxlen2)):
            #
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            #y_hat : [batch_size, senten_len] 其中的元素是翻译的词对应的index

            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        #监控一个eval值
        #tensorflow.random_unifrom 第一个参数是shape, 不填的话是一个数
        #第二个参数是下限，第三个参数是上限，第四个参数类型
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        #sent1 原始数据
        sent1 = sents1[n]
        #pred将
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        #sent2 预测数据
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

