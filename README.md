# Face-Alignment-DRF
# To be determined
* NME Loss 使用的 oculr distance 用预测点还是 ground truth.
* NME Loss 使用的 pupil distance 的眼睛中心点坐标如何计算。
* PCA 是否是模型精度的天花板，即模型最精确也只能达到 PCA 的误差？
* Test set 是否需要切割人脸。
* Train set 切割人脸采用 landmarks 坐标，导致很多点在切割之后的边缘，是否需要限制长宽比之类，修正切割框的位置。
# Questions
* 感觉不需要 Iterative training, 先训 gaussian 再训 CNN 即可？
# 日志
<details>
<summary>  <b> 5.26-5.30 </b > </summary>
  
## 5.26 周三 (6.5h)
* 发现原始的 landmarks 坐标经过 Procuses 变换后丧失了缩放、旋转、位移的数值，导致跟图片无法对应。重新做数据处理，直接用 PCA, 然后归一化到 (0,1) 之间。保证与图片对应。
  
## 5.27 周四 (7.5h)
* 日志早就应该开始写了，把每天遇到的问题或者想法记录下来，比草稿纸更有效。算了，从今天开始也不晚。
* 增加 GPU 部分，CNN 可以在 CUDA 上跑了。
* 调通了训练部分代码，但 Loss 基本没动，CNN 输出很小，做 Loss 的时候基本是 Mean Face.
* 之前的 Flag, 五月底之前至少有个结果，达成。即便结果很烂，但模型框架有了。
  
## 5.28 周五 (7.5h)
* 多变量高斯求 pdf 的函数原来用的是 scipy 库，但它只能 cpu 运行，且不支持输入矩阵，所以只能用两层循环，很慢。有多变量高斯的库很多，但基本是从分布里抽取随机数，不支持输入向量返回概率值。找到 torch.distributions 里有替代品，现在整个模型都能在 GPU 上跑了。
* 应该在开头定义 device 全局变量，免得一个变量一个变量的搬运到 cpu 或者 cuda。
* 模型训不动的问题，我觉得可能是这样，数据点的分布可能是很稀疏的。用 8 个 20 维高斯来拟合这些点的分布，首先需要很好的初始化，不然初始化到没有散点的空间里就会导致概率为 0，报错 "不能为NaN"之类。如果初始化时候这八个高斯差别不大，又会导致他们趋于同一个分布，无法向八个方向发展，变成用一个 20 维高斯来拟合。尝试解决：先用 3-5 维高斯试试能不能不那么依赖初始化的数值。
* 另一个之前没考虑过的是，如果拟合 8 个 20 维高斯，需要多少数据点。恐怕需要大量数据。
* 将多变量高斯初始化时的 Mean 设为 Kmeans 聚类中心点。
* 发现 CNN 输出一直很小，尝试把数据缩减为 2 维，用 EM 算法使多元正太收敛。缩减为二维以后可以画散点图帮助 debug. 结果证明，即便是二维的情况也无法收敛。仔细检查 EM 算法，没有问题。发现乘以了系数 pi 导致点全部缩到原点。CNN 输出一直很小的原因查明，解决方案待定。
  
## 5.29 周六 (0h)
* 吃饭睡觉，休息的一天。
* Gaussian 部分的 Inference 有问题, pi 的意义不对。

## 5.30 周日 (6h)
* 前几天一直遇到的问题是，CNN 的预测输出来是 Mean Shape, 今天得到解决。一个是 CNN 做 Loss 的方式不对，应该将 CNN 输出与这些每个样本放在 Multivariate-Gaussian 里面的得到的概率做 Loss，再一个是 CNN 训练不够，现在是 4000 Epoch 起（其实 2000 左右即可收敛，但具体多少跟 rf_dim 有关）。之前做 loss 用每个高斯的 Mean 乘以 CNN 输出（当作概率），一是输出没有归一化，导致很小，加上 Mean Shape 以后几乎被吞掉不计。二是没有发挥 Gaussian 的作用，训练出来的 Covariance 和 Pi 没有用上。
* 现在的问题是高斯维度没办法太高，太高会报错 Covariance 里有不合法值，导致预测误差很大。
* 跟老师聊了会天，可能思路要变，得换方法。
* 几篇文章要看，“Look at boundary""label distribution learning""Does Learning Specific Feature for Related Parts help" 以及想看的 Capsule Net 相关文章。
   
</code></pre>
</details>

## 5.31 周一 (xh)
* 今天主要任务是看文献。

## 6.01 周二 (xh)

<details>
<summary>  <b> 5.31-6.06 </b > </summary>

## 6.07 周一 (xh)


## 6.08 周二 (xh)

</code></pre>
</details>





