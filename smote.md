
本文将主要详细介绍一下SMOTE(Synthetic Minority Oversampling Technique)算法从原理到代码实践，SMOTE主要是用来解决类不平衡问题的，在讲解SMOTE算法之前，我们先解释一下什么是类不平衡问题、为什么类不平衡带来的问题以及相应的解决方法。


## 1. 什么是类不平衡问题

类不平衡（class-imbalance）是指在训练分类器中所使用的训练集的类别分布不均。比如说一个二分类问题，1000个训练样本，比较理想的情况是正类、负类样本的数量相差不多；而如果正类样本有995个、负类样本仅5个，就意味着存在类不平衡。

在后文中，把样本数量过少的类别称为“少数类”。

但实际上，数据集上的类不平衡到底有没有达到需要特殊处理的程度，还要看不处理时训练出来的模型在验证集上的效果。有些时候是没必要处理的。


## 2. 类不平衡引发的问题


### 2.1 从模型的训练过程来看

从训练模型的角度来说，如果某类的样本数量很少，那么这个类别所提供的“信息”就太少。

使用经验风险(模型在训练集上的平均损失)最小化作为模型的学习准则。设损失函数为0-1 loss(这是一种典型的均等代价的损失函数)，那么优化目标就等价于错误率最小化(也就是accuracy最大化)。考虑极端情况：1000个训练样本中，正类样本999个，负类样本1个。训练过程中在某次迭代结束后，模型把所有的样本都分为正类，虽然分错了这个负类，但是所带来的损失实在微不足道，accuracy已经是99.9%，于是满足停机条件或者达到最大迭代次数之后自然没必要再优化下去，ok，到此为止，训练结束！于是这个模型……

模型没有学习到如何去判别出少数类，这时候模型的召回率会非常低。



### 2.2 从模型的预测过程来看

考虑二项Logistic回归模型。输入一个样本 x ，模型输出的是其属于正类的概率 ŷ  。当 ŷ >0.5 时，模型判定该样本属于正类，否则就是属于反类。

为什么是0.5呢？可以认为模型是出于最大后验概率决策的角度考虑的，选择了0.5意味着当模型估计的样本属于正类的后验概率要大于样本属于负类的后验概率时就将样本判为正类。但实际上，这个后验概率的估计值是否准确呢？

从几率(odds)的角度考虑：几率表达的是样本属于正类的可能性与属于负类的可能性的比值。模型对于样本的预测几率为 $\frac{ŷ} {1−ŷ}$ 。

模型在做出决策时，当然希望能够遵循真实样本总体的正负类样本分布：设 θ 等于正类样本数除以全部样本数，那么样本的真实几率为 $\frac{θ}{1−θ}$ 。当观测几率大于真实几率时，也就是 $ŷ >θ$ 时，那么就判定这个样本属于正类。

虽然我们无法获悉真实样本总体，但之于训练集，存在这样一个假设：训练集是真实样本总体的无偏采样。正是因为这个假设，所以认为训练集的观测几率$\frac { \hat{\theta}}{1−\hat{ \theta }}$就代表了真实几率 $\frac{θ}{1−θ}$ 。

所以，在这个假设下，当一个样本的预测几率大于观测几率时，就应该将样本判断为正类。


## 3. 解决类不平衡问题的方法

目前主要有三种办法：

### 3.1 调整 θ 值

根据训练集的正负样本比例，调整 θ 值。   

这样做的依据是上面所述的对训练集的假设。但在给定任务中，这个假设是否成立，还有待讨论。


### 3.2 过采样

对训练集里面样本数量较少的类别（少数类）进行过采样，合成新的样本来缓解类不平衡。下面将介绍一种经典的过采样算法：SMOTE。

### 3.3 欠采样

对训练集里面样本数量较多的类别（多数类）进行欠采样，抛弃一些样本来缓解类不平衡。



## 4. SMOTE算法原理

SMOTE，合成少数类过采样技术．它是基于随机过采样算法的一种改进方案，由于随机过采样采取简单复制样本的策略来增加少数类样本，这样容易产生模型过拟合的问题，即使得模型学习到的信息过于特别(Specific)而不够泛化(General)，SMOTE算法的基本思想是对少数类样本进行分析并根据少数类样本人工合成新样本添加到数据集中，算法流程如下。

### 4.1 SMOTE算法流程

对于正样本数据集X(minority class samples)，遍历每一个样本：

$\,\,\,\,\,\,$   (1) 对于少数类(X)中每一个样本x，计算它到少数类样本集(X)中所有样本的距离，得到其k近邻。
    
$\,\,\,\,\,\,$   (2) 根据样本不平衡比例设置一个采样比例以确定采样倍率sampling_rate，对于每一个少数类样本x，从其k近邻中随机选择sampling_rate个近邻，假设选择的近邻为 ${x^{(1)}, x^{(2)}, ..., x^{(sampling\_rate)}}$ 。
    
$\,\,\,\,\,\,$   (3) 对于每一个随机选出的近邻 $x^{(i)} \,  (i=1,2, ..., {sampling\_rate}) $，分别与原样本按照如下的公式构建新的样本

$$ x_{new} = x + rand(0, 1) * (x^{(i)} - x) $$


### 4.2 SMOTE算法代码实现

下面我们就用代码来实现一下：


```python
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    """
    SMOTE过采样算法.


    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """
    def __init__(self, sampling_rate=5, k=5):
        self.sampling_rate = sampling_rate
        self.k = k
        self.newindex = 0

    def fit(self, X, y=None):
        if y is not None:
            negative_X = X[y==0]
            X = X[y==1]
            
        n_samples, n_features = X.shape
        # 初始化一个矩阵, 用来存储合成样本
        self.synthetic = np.zeros((n_samples * self.sampling_rate, n_features))
        
        # 找出正样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k).fit(X)
        for i in range(len(X)):
            k_neighbors = knn.kneighbors(X[i].reshape(1,-1), 
                                         return_distance=False)[0]
            # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成
            # sampling_rate个新的样本
            self.synthetic_samples(X, i, k_neighbors)
            
        if y is not None:
            return ( np.concatenate((self.synthetic, X, negative_X), axis=0), 
                     np.concatenate(([1]*(len(self.synthetic)+len(X)), y[y==0]), axis=0) )
        
        return np.concatenate((self.synthetic, X), axis=0)


    # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成sampling_rate个新的样本
    def synthetic_samples(self, X, i, k_neighbors):
        for j in range(self.sampling_rate):
            # 从k个近邻里面随机选择一个近邻
            neighbor = np.random.choice(k_neighbors)
            # 计算样本X[i]与刚刚选择的近邻的差
            diff = X[neighbor] - X[i]
            # 生成新的数据
            self.synthetic[self.newindex] = X[i] + random.random() * diff
            self.newindex += 1
            
X=np.array([[1,2,3],[3,4,6],[2,2,1],[3,5,2],[5,3,4],[3,2,4]])
y = np.array([1, 1, 1, 0, 0, 0])
smote=Smote(sampling_rate=1, k=5)
print(smote.fit(X))
```

    [[ 2.55355825  3.55355825  5.33033738]
     [ 3.          4.89432435  2.42270262]
     [ 2.          2.          1.        ]
     [ 3.          5.          2.        ]
     [ 5.          3.          4.        ]
     [ 3.          3.85514586  5.85514586]
     [ 1.          2.          3.        ]
     [ 3.          4.          6.        ]
     [ 2.          2.          1.        ]
     [ 3.          5.          2.        ]
     [ 5.          3.          4.        ]
     [ 3.          2.          4.        ]]


### 4.3 SMOTE算法的缺陷

该算法主要存在两方面的问题：一是在近邻选择时，存在一定的盲目性。从上面的算法流程可以看出，在算法执行过程中，需要确定k值，即选择多少个近邻样本，这需要用户自行解决。从k值的定义可以看出，k值的下限是sampling_rate(sampling_rate为从k个近邻中随机挑选出的近邻样本的个数，且有 sampling_rate < k ), sampling_rate的大小可以根据负类样本数量、正类样本数量和数据集最后需要达到的平衡率决定。但k值的上限没有办法确定，只能根据具体的数据集去反复测试。因此如何确定k值，才能使算法达到最优这是未知的。

另外，该算法无法克服非平衡数据集的数据分布问题，容易产生分布边缘化问题。由于正类样本(少数类样本)的分布决定了其可选择的近邻，如果一个正类样本处在正类样本集的分布边缘，则由此正类样本和相邻样本产生的“人造”样本也会处在这个边缘，且会越来越边缘化，从而模糊了正类样本和负类样本的边界，而且使边界变得越来越模糊。这种边界模糊性，虽然使数据集的平衡性得到了改善，但加大了分类算法进行分类的难度。

针对SMOTE算法的进一步改进

针对SMOTE算法存在的边缘化和盲目性等问题，很多人纷纷提出了新的改进办法，在一定程度上改进了算法的性能，但还存在许多需要解决的问题。

Han等人Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning 在SMOTE算法基础上进行了改进,提出了Borderhne.SMOTE算法,解决了生成样本重叠(Overlapping)的问题该算法在运行的过程中,查找一个适当的区域,该区域可以较好地反应数据集的性质,然后在该区域内进行插值,以使新增加的“人造”样本更有效。这个适当的区域一般由经验给定,因此算法在执行的过程中有一定的局限性。



参考文献：

http://www.cnblogs.com/Determined22/p/5772538.html

smote算法的论文地址：https://www.jair.org/media/953/live-953-2037-jair.pdf

http://blog.csdn.net/yaphat/article/details/60347968

http://blog.csdn.net/Yaphat/article/details/52463304?locationNum=7#0-tsina-1-78137-397232819ff9a47a7b7e80a40613cfe1

