# Face-Alignment-DRF
# To be determined
* NME Loss 使用的 oculr distance 用预测点还是 ground truth.
* NME Loss 使用的 pupil distance 的眼睛中心点坐标如何计算。
* PCA 是否是模型精度的天花板，即模型最精确也只能达到 PCA 的误差？
* Test set 是否需要切割人脸。
* Train set 切割人脸采用 landmarks 坐标，导致很多点在切割之后的边缘，是否需要限制长宽比之类，修正切割框的位置。
* 感觉不需要 Iterative training, 先训 gaussian 再训 CNN 即可？
# Weekly
| Time | Achievements |
|  ----  | ----  |
| **4.30-5.02** | 考完试休息两天，补觉，倒回时差。 |

**5.03-5.09**: 找房租房搬家，写大作业结课报告，补觉。

**5.10-5.16**: 读文献，看大家做 Face Alignment 用什么数据集、用什么指标评价、用什么方法。

**5.17-5.23**: 细读与我方法有关的文献。完成裁剪人脸、裁剪特征点、PCA，Procuses Analysis 代码。

**5.24-5.30**: 实现 Gaussian Regression Tree 方法 (Sequential)，得到初步结果。

**5.31-6.06**: 实现 Gaussian Regression Tree 方法 (Iterative)，结果误差很大且难收敛。搞清楚了数据集，全部下载到手。

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
* 吃饭睡觉，休息的一天。点评上说“镇鼎鸡”是上海老字号了，白切鸡专卖。刚好附近有一家，遂去吃。白切鸡做法简单，取三黄鸡水煮即可，熟度至刚刚断生为最佳，鸡肉全靠蘸料提味。难点在于，鸡肉不能有腥味，如果原材料不新鲜或者放久了都不行。这家店价格还行，买四分之一只（鸡腿部），加一碗鸡汁葱油拌面，五十块。如果店面离我更近，应该会经常去。
* Gaussian 部分的 Inference 有问题, pi 的意义不对。

## 5.30 周日 (6h)
* 前几天一直遇到的问题是，CNN 的预测输出来是 Mean Shape, 今天得到解决。一个是 CNN 做 Loss 的方式不对，应该将 CNN 输出与这些每个样本放在 Multivariate-Gaussian 里面的得到的概率做 Loss，再一个是 CNN 训练不够，现在是 4000 Epoch 起（其实 2000 左右即可收敛，但具体多少跟 rf_dim 有关）。之前做 loss 用每个高斯的 Mean 乘以 CNN 输出（当作概率），一是输出没有归一化，导致很小，加上 Mean Shape 以后几乎被吞掉不计。二是没有发挥 Gaussian 的作用，训练出来的 Covariance 和 Pi 没有用上。
* 现在的问题是高斯维度没办法太高，太高会报错 Covariance 里有不合法值，导致预测误差很大。
* 跟老师聊了会天，可能思路要变，得换方法。
* 几篇文章要看，“Look at boundary""label distribution learning""Does Learning Specific Feature for Related Parts help" 以及想看的 Capsule Net 相关文章。
   
</details>

<details>
<summary>  <b> 5.31-6.06 </b > </summary>

## 5.31 周一 (6.5h)
* 今天主要任务是看文献。
* 看"Age estimation", 深入公式，确实原来漏掉了一部分内容，但大差不差。怎么优化心里有数了，文章里边把连续随机变量的概率密度值乘以置信度得到另一个概率，但连续随机变量概率密度是不一定在 0-1 之间的，只是概率密度对随机变量的积分为 1. 离散随机变量的概率值才在 (0,1) 之间，所以这里需要归一化，但文章没写。这个点是文章里非常容易忽略，实践起来容易出错的东西，因为多元正太维度变高以后这个概率密度会变得巨大，以至于报错。另需要找找怎么对自定义损失函数用 Pytorch 自动求导。明天写代码，争取复现。
* 看"Learning Specific Feature", 跟预想的差不多，把互相联系的需要求的变量放在一组进行回归，模型可以少学一些不必要的变换，能提高精度。但他提到求解互信息的方法，以及如何将网络堆叠，是我没考虑到的。
* 看"Label Distributiob Learning", 标题说是预测分布，搞得我以为是得到一个函数，其实是为每个可能的 label 预测一个可能性罢了，叫 distribution. 做 Loss 的时候把 KL divergence 转化为 Cross entropy loss, Leaf Node 用 Variational Bounding, Split node 用 Back probagation. 没明白 label 由 one-hot 改为 distribution 有什么好处，可能不是这么改有好处，而是根据问题的实际意义，有的可以用 one-hot label, 有的需要用 distribution. 

## 6.01 周二 (7.5h)
* 按照昨天看 "Age detection" 的方法改代码，又出现所有图片的 probs 全一样的情况。很迷。输出完全没有因图而异，从理论分析我感觉是 loss 有问题。
* 代码着实码不出来，报错 Covariance matrix 有非法值，其实就是有的 Covariance Matrix 里的值小于 1e-6，导致被判定 sigma 为 Singular matrix...难道要开始推公式了吗...应该是更新参数的问题。
* Regression tree 的 train 部分检查过了，应该没问题了。解决所有图片输出一致的问题，尝试在第一轮训练让 cnn output 拟合各自图片在 gaussian 里的概率，在第一步训练用 L1 loss 让 cnn 输出各异，后续正常用 cross entropy loss。失败。每张图片输出还是一样，无法各自拟合标签。
* 解决每张图片输出相同的问题，与其说是解决，不如说是问题自己消失了。尝试增大 CNN 训练的 epoch，没用。改变 learning rate，没用。将 loss function 换成手写的，在数学上等价的函数，没用。怀疑使用了 in-place operation 导致 pytorch 建图错误，反复检测，没用。后来某次重启机器，顺利收敛。

## 6.02 周三 (7.5h)
* 奇了，昨天代码都没动，只是今天开机重新跑一遍，结果又出现 CNN 输出一样数据的问题。
* 除此之外还报错 Covariance 有 invalid value。尝试先用 svd，强行把过小的 singular value 改为 1e-5 以躲过正确性检查，但这样训下来有的 Covariance 居然变成 0 了。无计可施。
* invalid value 以强行打补丁的形式解决。还剩 CNN 输出一致的问题。
* 小了，格局小了。之前 Sequential Training 的时候，CNN 训练 400 epoch，不收敛，后来加到 2000 epoch, 发现在 500-800 epoch 的时候，loss 会迅速下降。即，loss 会先从 0.5 降到 0.2, 大概花几十个 epoch，然后一直维持在 loss=0.2 不动。训练到 500-800 epoch, loss 突然开始下降，很快收敛到 0 附近。所以在 Iterative training，直接给 epoch 设为 2000，结果是没有办法收敛。这是前言。训 Iterative Training 训不动，转头去看之前可以收敛的 Sequential Training，其实这段代码偶尔也不能收敛，所以一定有没有查明的问题。有一次训 Sequentian Training 的 CNN，发现中间的 loss 维持在平台期达到 1300 epoch，收到启发，在 Iterative Training 把 epoch 加到 1w，可惜还是不能收敛，几乎排除 epoch 不够大的原因。
* 左图：Sequential Training, 右图：Iterative Training.

![Sequential Training](figs/amazing.png)
![Iterative Training](figs/fail.png)

## 6.03 周四 (4h)
* 重新跑了昨天的代码，问题依旧。整理本周进展，做 PPT，备明日汇报。

## 6.04 周五 (4.5h)
* 昨夜失眠，三点才睡着。幸好不是社畜，不用明天八点上班。睡不着的原因应该是最近减肥，摄入不抵消耗，躺在床上很饿。晚饭在食堂吃了一碗云吞加一只鸭腿，又点了一份炒饭，已经吃这么多了，没想到晚上还是饿，三点爬起来摸出一袋饼干吃了，方才睡去。为什么要这么痛苦的节食减肥呢，瘦下来以后也不是到了终点，可以敞开吃喝。人生还得过，难道想保持身材就要一直节食吗，成本也太高了。
* CNN 输出一致的问题，我打算减小 learning rate 再试试，组会提出这个问题，被建议更换 optimizer, 之前用 Adam，那换成 SGDM 试试。实验结果：两个方法都没用。Loss 的平台现象仍然存在。

## 6.05 周六 (0.5h)
* 补觉。睡了13小时。

## 6.06 周日 (2h)
* 看书三分之一本，照此进度有望本月看完。
  
</details>

<details>
<summary>  <b> 6.07-6.13 </b > </summary>
## 6.07 周一 (8h)
* 收敛了，代码完全没改过。来实验室第一件事，把上次的代码跑一遍看看。周末没想到还有什么可能的原因导致 CNN 输出一致，就给自己放了两天假，结果今天问题消失了。Adam 优化器，lr: 3e-4，其实这组参数上周五试过，不行，但今天就有了。神奇。把结果保存下来。本来已经做好了搞体力活的准备，挨个检查输出是不是跟手算的一样，检查梯度啥的，看来不用了。但稳定性为什么会是个问题，原因尚不明确，代码不能复现也不行啊。

![Iterative Training](figs/cnn2_train_successful.png)

* 学到一个小知识，arXiv 读 archive.
* 发现 test label 切的不齐，数据处理的代码需要修改。
* 又跑了几遍代码，都能收敛。但总的来说模型学不到太多变化，打算采用分而治之的方式，选取有联系的点单独学习，而不是胡子眉毛一把抓。
* 换 Heatmap Regression 解决人脸对齐，完成：由 landmarks 坐标生成热图，训练 regression tree，训练 CNN 的代码。代码里的问题还有很多，明天仔细调整。

## 6.08 周二 (6.5h)
* 改完 Heatmap Regression 训练部分代码，现在能输出比较像样的热图。下一步，从热图找到特征点坐标。直接 topk() 存在全局最大无法代表局部最大的问题，如果用动态阈值扫描，还需要考虑两个点相距很近的情况。
* 想到，即便找出特征点坐标，也没法排序。不排序就不知道哪个点是眼角，无法计算 NME Loss。这才明白，为什么人家都是每个特征点做一个 Heatmap. 既能减小预测误差，又能知道顺序。

## 6.09 周三 (9h)
* 把 Heatmap 由所有点一张图改为每个点一张图，写代码。这样还有个好处，就是如果后面需要把相关性强的点分组回归，自然需要一个点一张 Heatmap.
* 改为一个点一张 Heatmap 以后计算量陡增，难以收敛。且学习 Distribution 似乎失去了意义，因为只有一张图只有一个目标点，变成 one-hot label.
* 观察到不同 Leaf node 差别不大，cnn 输出也在 0.5 附近，可知基本无筛选，算法并未收敛。重新研究如何优化 leaf node，采用 pinv 还是 step by step. 

## 6.10 周四 (7.5h)
* 今天是 Math Day. 1) KL Divergence & Entropy & Cross Entropy; 2) Linear Least Square & Pseudo Inverse; 3) Jensen's Inequality; 4) Convex Optimization & Lagrangian & Duality.

## 6.11 周五 (5h)
* 组会，提出几个新思路，后续改进。1）backbone 用 pretrain 人脸检测 model。2）backbone 用 Hourglass。 3）CNN FC 直接出 Heatmap 看看能否收敛。

## 6.12 周六 (0h)
* 补觉

## 6.13 周日 (0h)
* 补觉

</details>

## 6.14 周一 (xh)


## 6.15 周二 (xh)



<details>
<summary>  <b> 6.21-6.27 </b > </summary>
## 6.21 周一 (xh)


## 6.22 周二 (xh)


</details>





