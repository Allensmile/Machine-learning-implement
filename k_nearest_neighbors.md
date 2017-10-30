
# K近邻分类算法 (K-Nearest Neighbor) 


KNN分类算法非常简单，该算法的核心思想是如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别。该方法在确定分类决策上只依据最邻近K个样本的类别来决定待分样本所属的类别。KNN是一个懒惰算法，也就是说在平时不好好学习，考试（对测试样本分类）时才临阵发力（临时去找k个近邻），因此在预测的时候速度比较慢。

KNN模型是非参数模型，既然有非参数模型，那就肯定还有参数模型，那何为参数模型与非参数模型呢？



## 1. 参数模型与非参数模型

### 1.1 参数模型

参数模型是指选择某种形式的函数并通过机器学习用一系列固定个数的参数尽可能表征这些数据的某种模式。参数模型具有如下特征：
1. 不管数据量有多大，在模型确定了，参数的个数就确定了，即参数个数不随着样本量的增大而增加，从关系上说它们相互独立；
2. 一般参数模型会对数据有一定的假设，如分布的假设，空间的假设等，并且这些假设可以由参数来描述；
3. 参数模型预测速度快。

常用参数学习的模型有： 

* 回归模型（线性回归、岭回归、lasso回归、多项式回归）
* 逻辑回归
* 线性判别分析（Linear Discriminant Analysis）
* 感知器
* 朴素贝叶斯
* 神经网络
* 使用线性核的SVM
* Mixture models
* K-means
* Hidden Markov models
* Factor analysis / pPCA / PMF


### 1.2 非参数模型

非参数模型是指系统的数学模型中非显式地包含可估参数。注意不要被名字误导，非参不等于无参。非参数模型具有以下特征：

1. 数据决定了函数形式，函数参数个数不固定；
2. 随着训练数据量的增加，参数个数一般也会随之增长，模型越来越大；
3. 对数据本身做较少的先验假设；
4. 预测速度慢。

一些常用的非参学习模型： 

* k-Nearest Neighbors
* Decision Trees like CART and C4.5
* 使用非线性核的SVM
* Gradient Boosted Decision Trees
* Gaussian processes for regression
* Dirichlet process mixtures
* infinite HMMs
* infinite latent factor models


## 2. KNN算法步骤：

1. 准备数据，对数据进行预处理；

2. 设定参数，如k；

3. 遍历测试集，

      对测试集中每个样本，计算该样本（测试集中）到训练集中每个样本的距离；

      取出训练集中到该样本（测试集中）的距离最小的k个样本的类别标签；

      对类别标签进行计数，类别标签次数最多的就是该样本（测试集中）的类别标签。


4. 遍历完毕.





## 3. KNN算法优点和缺点

### 3.1 KNN算法优点

1. 简单，易于理解，易于实现，无需估计参数，无需训练；

2. 适合对稀有事件进行分类；

3. 特别适合于多分类问题(multi-modal,对象具有多个类别标签)， kNN比SVM的表现要好；

4. 由于KNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合；

5. 该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算法比较容易产生误分。



### 3.2 KNN算法缺点

1. 该算法在分类时有个主要的不足是，当样本不平衡时，如一个类的样本容量很大，而其他类样本容量很小时，有可能导致当输入一个新样本时，该样本的K个邻居中大容量类的样本占多数。 该算法只计算“最近的”邻居样本，某一类的样本数量很大，那么或者这类样本并不接近目标样本，或者这类样本很靠近目标样本。无论怎样，数量并不能影响运行结果。可以采用权值的方法（和该样本距离小的邻居权值大）来改进；

2. 该方法的另一个不足之处是计算量较大，因为对每一个待分类的文本都要计算它到全体已知样本的距离，才能求得它的K个最近邻点。

3. 属于硬分类，即直接给出这个样本的类别，并不是给出这个样本有多大的可能性属于该类别；

4. 可解释性差，无法给出像决策树那样的规则；

5. 计算量较大。目前常用的解决方法是事先对已知样本点进行剪辑，事先去除对分类作用不大的样本。



```python
from __future__ import print_function
import math
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
%matplotlib inline


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
        
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    
    return X[idx], y[idx]



# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 标准化数据集 X
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # 做除法运算时请永远记住分母不能等于0的情形
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0) 
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

    return X_std


# 划分数据集为训练集和测试集
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0] * (1-test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test

def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y == y_pred)/len(y)


class KNN():
    """ K近邻分类算法.

    Parameters:
    -----------
    k: int
        最近邻个数.
    """
    def __init__(self, k=5):
        self.k = k

    # 计算一个样本与训练集中所有样本的欧氏距离的平方
    def euclidean_distance(self, one_sample, X_train):
        one_sample = one_sample.reshape(1, -1)
        X_train = X_train.reshape(X_train.shape[0], -1)
        distances = np.power(np.tile(one_sample, (X_train.shape[0], 1)) - X_train, 2).sum(axis=1)
        return distances
    
    # 获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        k_neighbor_labels = []
        for distance in np.sort(distances)[:k]:

            label = y_train[distances==distance]
            k_neighbor_labels.append(label)

        return np.array(k_neighbor_labels).reshape(-1, )
    
    # 进行标签统计，得票最多的标签就是该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train, k):
        distances = self.euclidean_distance(one_sample, X_train)
        y_train = y_train.reshape(y_train.shape[0], 1)
        k_neighbor_labels = self.get_k_neighbor_labels(distances, y_train, k)
        
        find_label, find_count = 0, 0
        for label, count in Counter(k_neighbor_labels).items():
            if count > find_count:
                find_count = count
                find_label = label
        return find_label
    
    # 对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        y_pred = []
        for sample in X_test:
            label = self.vote(sample, X_train, y_train, self.k)
            y_pred.append(label)
        return np.array(y_pred)


def main():
    data = make_classification(n_samples=200, n_features=4, n_informative=2, 
                               n_redundant=2, n_repeated=0, n_classes=2)
    X, y = data[0], data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    
    accu = accuracy(y_test, y_pred)
    print ("Accuracy:", accu)


if __name__ == "__main__":
    main()

```

    Accuracy: 0.893939393939

