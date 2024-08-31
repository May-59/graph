任务一、 节点分类
实现基于GNN主流模型(GCN, GAT, GraphSAGE, GIN)的节点分类:

实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图上的节点分类任务
使用Cora, Citeseer, Flickr数据集
测试GCN, GAT, GraphSAGE, GIN模型
利用框架自带的Sampler采样子图进行训练，并与全图训练进行性能和运行时间的对比

任务二、 图上的链路预测
实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图上的链路预测
使用Cora, Citeseer, Flickr数据集
测试GCN, GAT, GraphSAGE, GIN模型
利用框架自带的Sampler采样子图进行训练，并与全图训练进行性能和运行时间的对比

任务三、 图分类
实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图分类任务
使用TUDataset, ZINC数据集
分析不同的池化方法对图分类性能的影响(AvgPooling, MaxPooling, MinPooling)
测试GCN, GAT, GraphSAGE, GIN模型

任务四、 知识图谱
参考
训练和测试框架KGE框架
TransE, RotatE, ConvE的论文
实现要求：基于参考资料和知识图谱补全框架，支持常见模型(TransE, RotatE, ConvE)
需要了解的知识点：
常见知识图谱补全模型的原理(TransE, RotatE, ConvE)
数据集：训练集/验证集/测试集的划分
