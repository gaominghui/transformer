#### attention
An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key.  

attenion机制中的output的计算可以看做是values的加权累加，而权重的计算是跟query及其对英国的key有关。  

在attention计算中，Q*K后，需要除以一个数，在做softmax之前，这样做的原因是因为，矩阵相乘后可能会得到一个比较大的值，如果不除，在做softmax时会非常接近于1或者0，
那么得到的梯度会非常小。所以需要除以维度的开平方