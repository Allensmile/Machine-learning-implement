

朴素贝叶斯算法是基于贝叶斯定理和特征之间条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入/输出的联合概率分布；然后基于此模型，对给定的输入x，利用贝叶斯定理求出后验概率最大的输出y。朴素贝叶斯算法实现简单，学习和预测的效率都很高，是一种常用的方法。

本文考虑当特征是连续情形时，朴素贝叶斯分类方法的原理以及如何从零开始实现贝叶斯分类算法。此时，我们通常有两种处理方式，第一种就是将连续特征离散化(区间化)，这样就转换成了离散情形，完全按照特征离散情形即可完成分类，具体原理可以参见上一篇文章。第二种处理方式就是本文的重点，详情请看下文：





## 1. 朴素贝叶斯算法原理


与特征是离散情形时原理类似，只是在计算后验概率时有点不一样，具体计算方法如下：

这时，可以假设每个类别中的样本集的每个特征均服从正态分布，通过其样本集计算出均值和方差，也就是得到正态分布的密度函数。有了密度函数，就可以把值代入，算出某一点的密度函数的值。为了阐述的更加清楚，下面我摘取了一个实例，以供大家更好的理解。



## 2. 朴素贝叶斯的应用


下面是一组人类身体特征的统计资料。

    　　性别　　身高（英尺）　体重（磅）　　脚掌（英寸）

    　　男 　　　6 　　　　　　180　　　　　12
    　　男 　　　5.92　　　　　190　　　　　11
    　　男 　　　5.58　　　　　170　　　　　12
    　　男 　　　5.92　　　　　165　　　　　10
    　　女 　　　5 　　　　　　100　　　　　6
    　　女 　　　5.5 　　　　　150　　　　　8
    　　女 　　　5.42　　　　　130　　　　　7
    　　女 　　　5.75　　　　　150　　　　　9

已知某人身高6英尺、体重130磅，脚掌8英寸，请问该人是男是女？

根据朴素贝叶斯分类器，计算下面这个式子的值。

    P(身高|性别) x P(体重|性别) x P(脚掌|性别) x P(性别)

这里的困难在于，由于身高、体重、脚掌都是连续变量，不能采用离散变量的方法计算概率。而且由于样本太少，所以也无法分成区间计算。怎么办？

这时，可以假设男性和女性的身高、体重、脚掌都是正态分布，通过样本计算出均值和方差，也就是得到正态分布的密度函数。有了密度函数，就可以把值代入，算出某一点的密度函数的值。

比如，男性的身高是均值5.855、方差0.035的正态分布。所以，男性的身高为6英尺的概率的相对值等于1.5789（大于1并没有关系，因为这里是密度函数的值，只用来反映各个值的相对可能性）。

从上面的计算结果可以看出，分母都一样，因此，我们只需要比价分子的大小即可。显然，P(不转化|Mx上海)的分子大于P(转化|Mx上海)的分子，因此，这个上海男性用户的预测结果是不转化。这就是贝叶斯分类器的基本方法：在统计资料的基础上，依据某些特征，计算各个类别的概率，从而实现分类。

$$ p(height|male) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(6-\mu)^2}{2 \sigma^2}} \approx 1.5789 $$

有了这些数据以后，就可以计算性别的分类了。

    　　P(身高=6|男) x P(体重=130|男) x P(脚掌=8|男) x P(男)
    　　　　= 6.1984 x e-9

    　　P(身高=6|女) x P(体重=130|女) x P(脚掌=8|女) x P(女)
    　　　　= 5.3778 x e-4

可以看到，女性的概率比男性要高出将近10000倍，所以判断该人为女性。


```python
from __future__ import division, print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
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




class NaiveBayes():
    """朴素贝叶斯分类模型. """
    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        # 存储高斯分布的参数(均值, 方差), 因为预测的时候需要, 模型训练的过程中其实就是计算出
        # 所有高斯分布(因为朴素贝叶斯模型假设每个类别的样本集每个特征都服从高斯分布, 固有多个
        # 高斯分布)的参数
        self.parameters = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        # 计算每一个类别每个特征的均值和方差
        for i in range(len(self.classes)):
            c = self.classes[i]
            # 选出该类别的数据集
            x_where_c = X[np.where(y == c)]
            # 计算该类别数据集的均值和方差
            self.parameters.append([])
            for j in range(len(x_where_c[0, :])):
                col = x_where_c[:, j]
                parameters = {}
                parameters["mean"] = col.mean()
                parameters["var"] = col.var()
                self.parameters[i].append(parameters)

    # 计算高斯分布密度函数的值
    def calculate_gaussian_probability(self, mean, var, x):
        coeff = (1.0 / (math.sqrt((2.0 * math.pi) * var)))
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return coeff * exponent

    # 计算先验概率 
    def calculate_priori_probability(self, c):
        x_where_c = self.X[np.where(self.y == c)]
        n_samples_for_c = x_where_c.shape[0]
        n_samples = self.X.shape[0]
        return n_samples_for_c / n_samples

    # Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
    # P(X|Y) - Probability. Gaussian distribution (given by calculate_probability)
    # P(Y) - Prior (given by calculate_prior)
    # P(X) - Scales the posterior to the range 0 - 1 (ignored)
    # Classify the sample as the class that results in the largest P(Y|X)
    # (posterior)
    def classify(self, sample):
        posteriors = []
        
        # 遍历所有类别
        for i in range(len(self.classes)):
            c = self.classes[i]
            prior = self.calculate_priori_probability(c)
            posterior = np.log(prior)
            
            # probability = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            # 遍历所有特征 
            for j, params in enumerate(self.parameters[i]):
                # 取出第i个类别第j个特征的均值和方差
                mean = params["mean"]
                var = params["var"]
                # 取出预测样本的第j个特征
                sample_feature = sample[j]
                # 按照高斯分布的密度函数计算密度值
                prob = self.calculate_gaussian_probability(mean, var, sample_feature)
                # 朴素贝叶斯模型假设特征之间条件独立，即P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y), 
                # 并且用取对数的方法将累乘转成累加的形式
                posterior += np.log(prob)
            
            posteriors.append(posterior)
        
        # 对概率进行排序
        index_of_max = np.argmax(posteriors)
        max_value = posteriors[index_of_max]

        return self.classes[index_of_max]

    # 对数据集进行类别预测
    def predict(self, X):
        y_pred = []
        for sample in X:
            y = self.classify(sample)
            y_pred.append(y)
        return np.array(y_pred)


def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = np.array(clf.predict(X_test))

    accu = accuracy(y_test, y_pred)

    print ("Accuracy:", accu)

    
if __name__ == "__main__":
    main()

```

    Accuracy: 0.973333333333


参考文献：
http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html

李航《统计学习方法》
