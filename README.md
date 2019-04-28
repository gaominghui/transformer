# **[UPDATED]** A TensorFlow Implementation of [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

When I opened this repository in 2017, there was no official code yet.
I tried to implement the paper as I understood, but to no surprise
it had several bugs. I realized them mostly thanks to people who issued here, so
I'm very grateful to all of them. Though there is the [official implementation](https://github.com/tensorflow/tensor2tensor) as well as
several other unofficial github repos, I decided to update my own one.
This update focuses on:
* readable / understandable code writing
* modularization (but not too much)
* revising known bugs. (masking, positional encoding, ...)
* updating to TF1.12. (tf.data, ...)
* adding some missing components (bpe, shared weight matrix, ...)
* including useful comments in the code.

I still stick to IWSLT 2016 de-en. I guess if you'd like to test on a big data such
as WMT, you would rely on the official implementation.
After all, it's pleasant to check quickly if your model works.
The initial code for TF1.2 is moved to the [tf1.2_lecacy](tf1.2_legacy) folder for the record.

## Requirements
* python==3.x (Let's move on to python 3 if you still use python 2)
* tensorflow==1.12.0
* numpy>=1.15.4
* sentencepiece==0.1.8
* tqdm>=4.28.1

## Training
* STEP 1. Run the command below to download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en).
```
bash download.sh
```
 It should be extracted to `iwslt2016/de-en` folder automatically.
* STEP 2. Run the command below to create preprocessed train/eval/test data.
```
python prepro.py
```
If you want to change the vocabulary size (default:32000), do this.
```
python prepro.py --vocab_size 8000
```
It should create two folders `iwslt2016/prepro` and `iwslt2016/segmented`.

* STEP 3. Run the following command.
```
python train.py
```
Check `hparams.py` to see which parameters are possible. For example,
```
python train.py --logdir myLog --batch_size 256 --dropout_rate 0.5
```

* STEP 3. Or download the pretrained models.
```
wget -qO- --show-progress https://dl.dropbox.com/s/efv2gmq5hu3np43/log.tar.gz | tar xz
```


## Training Loss Curve
<img src="fig/loss.png">

## Learning rate
<img src="fig/lr.png">

## Bleu score on devset
<img src="fig/bleu.png">


## Inference (=test)
* Run
```
python test.py --ckpt log/1/iwslt2016_E17L2.78-26078 (OR yourCkptFile OR yourCkptFileDirectory)
```

## Results
* Typically, machine translation is evaluated with Bleu score.
* All evaluation results are available in [eval/1](eval/1) and [test/1](test/1).

|tst2013 (dev) | tst2014 (test) |
|--|--|
|26.69|22.46|

## Notes
* Beam decoding will be added soon.
* I'm going to update the code when TF2.0 comes out if possible.


# 1流程讲解
文件阅读顺序：
download.sh  &rarr; prepo.py  &rarr; data_load.py &rarr; train.py  &rarr; test.py

参数文件和工具文件分别是 hparams.py 和 utils.py  
#### download.sh  
1.新建路径  
2.下载训练文件  
3.解压`数据会写到iwslt2016/de-en`       
#### prepo.py   
1.根据不同数据集的格式，使用不同的正则清洗和读入数据  
2.将1清洗后的数据写回到磁盘  `数据会写到iwslt2016/prepro`  
3.使用goolge开源的sentencepiece 训练源语言和目标语言的训练的合集，得到模型文件和词典文件  
4.使用3中的模型和词典文件，将所有数据集转换，并写回到磁盘`数据会写到iwslt2016/segmented`    
#### data_load.py   
1.载入字典，生成word &rarr; index的映射，index &rarr; word的映射  
2.过滤超过一定长度的句子  
3.转换原始句子为index, [word1,word2......] &rarr;[index1,index2......]  
4.生成batch  
需要说明的两个细节    
+ transformer在训练的时候使用了上下文的信息，在评估和预测的时候只能使用上文预测的信息。
 与此对应，目标语言在load的时候会有两种格式，一种会在训练的时候用，一种会在预测的时候用。

+ 为了保证每个batch里句子的长度一样，使用了padded_batch 方法，该方法会让同一个batch里所有
句子后边加入适当的0，来保证跟最长的句子一样长。而不同batch里句子长度不一定相同。
#### train.py
1.初始化数据，初始化模型，定义各种计算所需的operation  
2.初始化checkpoint,summary  
3.以一个batch为单位，进行训练，每个epoch ,也就是整个训练集训练一遍，做一次eval  
4.eval的时候 计算训练集的loss, 计算验证集的bleu，并将结果写回到磁盘。

### test.py
1.初始化预测及，初始化模型，定义各种计算所需的operation  
2.通过checkpoint load session和模型  
3.同train.py中做验证一样仍然用model中的eval作预测，并计算bleu

# 2模型讲解
主要有两个文件model.py和modules.py  
#### model.py
##### \__init\__   
transformer模型到构造文件,有如下若干个变量  
+ hp 超参数文件，学习率，batch_size,num_epochs等
+ token2idx word到index的映射
+ idx2token index到word的映射
+ embeddings shape : [vacal_szie,d_model]  vacal_szie:词典大小 d_model 词向量维度

##### encode  
1.embedding_lookup 为每个句子找到对应的隐向量[batch_size,T1] &rarr;[batch_size,T1,d_model]
T1:源语言句子长度 d_model: 词向量维度 batch_size:句子个数  
2.scale 论文中好像没有这个，不是特别清楚scale的作用
3.position_encoding
位置编码的计算只跟位置有关，而且这个可以做成类似于字典文件，只需计算一次，而不是每次都计算，
因为positon_encoding，计算的是最大长度的位置编码，然后通过embedding_lookup找对应位置的编码
4.encode_blocks,循环num_blocks次   
4.1 多头self-attention
+ 初始化Q,K,V 通过全连接计算 Q[batch_size,Tq,d_model] K[batch_size,Tk,d_model] V[batch_size,Tk,d_model] 
+ Q,K,V 多头转换： Q[batch_size*num_heads,Tq,d_model/num_heads] K[batch_size*num_heads,Tk,d_model/num_heads] V[batch_size*num_heads,Tk,d_model/num_heads]
+ 计算 z = softmax( scale(Q*K) )*V  Z[batch_size*num_heads,Tq,d_model/num_heads]
+ Z shape 转换： [batch_size,Tq,d_model]
+ 残差结构：Z+ Q
+ 多头attention的物理含义的理解:  softmax( scale(Q*K) )找与Q最相似的K,再乘以K,表示用k向量来表示Q向量，V向量好像有点冗余？
而多头的含义是两个较长的向量计算相似度往往不会很相似，拆成若干个子向量计算，会相似的概率会提升不少
+ 论文中说多头从两方面提升了模型的效果：
4.2 position-wise feed forward net
三层的全连接网络，含有残差结构

##### decode  
1.embedding_lookup 同encode  
2.scale   同encode  
3.position_encoding 同encode  
4.decode_blocks ,循环循环num_blocks次   
4.1 多头self-attention 同encode  
4.2 多头 encoder-decoder-attention  
与self attention 不同的地方在于 k,v是encocer的输出  
4.3 position-wise feed forward net 
##### train
训练模型
执行encode  
执行decode  
执行label_smooth，将index 转换成one-hot形式，然后做平滑,0变成一个略微比0大的值，1变成一个略微比1小的值  
执行softmax_cross_entropy_with_logits_v2  
去掉sentence中padding的部分  
其他的部分计算loss  
定义优化器  
定义学习率  
定义train_operation 优化器最优化loss
##### eval
评估模型
读入目标语言的句子时，将其替换为零向量，因为这个是做预测的  
然后逐字预测
这个跟mask方法有较大关系，目前还不是特别明白

#### modules.py
##### ln
layer normalization函数，减去均值，除以标准差
get_token_embeddings 获取word_embedding词典
scaled_dot_product_attention 计算attention的辅助函数
mask 还不是很明白
multihead_attention 多头attention
ff feed-forward network encode和decode中会用到
label_smoothing 将index 转换成one-hot形式，然后做平滑,0变成一个略微比0大的值，1变成一个略微比1小的值
positional_encoding 位置编码
noam_scheme 学习率的变化由小到大，再由大到小
