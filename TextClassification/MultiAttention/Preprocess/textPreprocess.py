'''
@Project:NLPAction

@Author:lincoln

@File:textPreprocess

@Time:2020-05-07 13:49:45

@Description:
'''
# 构建Embedding矩阵
# embedding矩阵是一个max_word行dim列的矩阵,代表着这max_word个词中每个词对应dim维向量
# Embedding层:input_shape(batch_size, input_length)
#             output_shape(bach_size，input_length, dim)
# 可以知道Embedding层是一个查表过程。比如batch_size = 2,input_length = 3
# 某一次输入[[100,2,20]，[12,34,11]]表示取出embedding[100]、embedding[2]、
# embedding[20]、embedding[12]、embedding[34]、embedding[11]
# 得[[embedding[100]、embedding[2]、embedding[20]],
#  [embedding[12]、embedding[34]、embedding[11]]]
# 其中embedding[100]等都为dim向量
# 另外需要注意的是embedding[0]是零向量因为索引为0的
# 有了上述了解后我们来构建embedding矩阵
import os
import numpy as np
def getEmbeddingMatrix(max_word, embedding_dim, word_index, embedding_dic):
    '''
    parameter:
        max_word:词汇表单词数目
        embedding_dim:词向量维度
        word_index: word to index
        embedding_dic: word to vector
    return:
        index to vector
    '''
    embedding_matrix = np.zeros((max_word, embedding_dim))
    for word, index in word_index.items():
        if index < max_word:
            embedding_vector = embedding_dic.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix


#读取词向量文件，返回词向量字典
def getWordVecDict(path,dim):
    embedding_dict = {}
    file_name = "glove.6B."+str(dim)+"d.txt"
    f = open(os.path.join(path,file_name))
    for line in f:
        values = line.split()
        word = values[0]
        vector = values[1:]
        embedding_dict[word] = vector
    f.close()
    return embedding_dict