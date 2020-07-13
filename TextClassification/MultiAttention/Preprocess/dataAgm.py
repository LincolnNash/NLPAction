#NLP数据增强
# 利用互信息做数据增强
#
from tensorflow.keras.datasets import imdb
import math

#计算香农熵
def calEntropy(labels):
    """
    :param labels: 类别标签
    :return: 香农熵
    """
    length = len(labels)
    entropy = 0.0
    categories_num = {}
    for category in labels:
        if category not in categories_num:
            categories_num[category] = 0
        categories_num[category] += 1
    for key, value in categories_num.items():
        entropy -= (value/length) * math.log(value/length, 2)
    return entropy

#计算互信息
def calMI(datasets, labels, y):
    """
    :param datasets: 数据集
    :param labels: 类别
    :param y: 特征
    :return: 互信息
    """
    length = len(labels)
    entropy_base = calEntropy(labels)

    ds_with_y = []
    ds_without_y = []
    for i in range(length):
        if y in datasets[i]:
            ds_with_y.append(labels[i])
        else:
            ds_without_y.append(labels[i])
    conditional_entropy = (len(ds_with_y)/length * calEntropy(ds_with_y)) + \
                          (len(ds_without_y)/length * calEntropy(ds_without_y))
    return entropy_base - conditional_entropy

#计算每个词的互信息
def getIndexToMI(datasets, labels, word_index):
    index_mi = {}
    for index in word_index.values():
        index_mi[index] = calMI(datasets, labels, index)
    return index_mi

#保存文件
def save_list(filename, data):#filename为写入文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("save_list successfully")


def save_dict(filename, data):
    file = open(filename,'w')
    for key, value in data.items():
        s = str(key)+":" + str(value)+"\n"
        file.write(s)
    file.close()
    print("save_dict successfully")

# 加载保存的字典文件
def load_dict(filename):
    file = open(filename,'r')
    index_mi = {}
    for item in file.readlines():
        s = item.replace("\n",'')
        s = s.split(":")
        index_mi[int(s[0])] = s[1]
    return index_mi

# 加载保存的list
def load_list(filename):
    file = open(filename,'r')
    datasets = []
    for item in file.readlines():
        sentence = []
        s = item.replace("\n",'')
        s = s.split()
        for i in s:
            sentence.append(int(i))
        datasets.append(sentence)
    return datasets

#数据增强
def dataAgm(datasets, index_mi):
    """
    :param datasets: 数据集
    :param index_mi: 互信息字典
    :return: 增强数据集
    """
    datasets_rep = []
    datasets_del = []
    for i in range(len(datasets)):
        if len(datasets[i])>20:
            data = datasets[i].copy()
            data_del = datasets[i].copy()
            data_rep = datasets[i].copy()
            for j in range(len(data)):
                data[j] = index_mi[data[j]]
            for _ in range(10):
                min_index = data.index(min(data))
                data_del.remove(data_del[min_index])
                data_rep[min_index] = 0
                data.remove(data[min_index])
        datasets_del.append(data_del)
        datasets_rep.append(data_rep)

    return datasets_del, datasets_rep

# 特征筛选：
def FeaSelect(data, index_mi):
    new_data = []

    for sentence in data:
        keeplen = math.floor(len(sentence / 2))
        new_sentence = []
        for index in sentence:
            mi_sentence=[]
            mi_sentence.append(index_mi[index])
            mi_sentence_sorted = sorted(mi_sentence)
            new_sentence = new_sentence[:keeplen]


# (x_train, y_train),(x_test,y_test) =imdb.load_data(path="../datasets/imdb/imdb.npz",num_words=10000,maxlen=400)
# print(x_train.shape,x_test.shape)
# word_index = imdb.get_word_index("../datasets/imdb/imdb_word_index.json")
# index_mi = load_dict("../datasets/imdb/index_mi.txt")
# datasets_del, datasets_rep = dataAgm(x_train,index_mi)
# save_list("../datasets/imdb/datasets_rep.txt", datasets_rep)
# save_list("../datasets/imdb/datasets_del.txt", datasets_del)



