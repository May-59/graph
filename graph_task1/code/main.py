from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import NormalizeFeatures
from models import GCN, GAT, GraphSAGE, GIN
import torch.nn.functional as F
import torch
import train
import test
import time 
from torch_geometric.loader import NeighborSampler


# 加载数据集

# 加载Cora数据集
# dataset = Planetoid(root='./dataset', name="Cora", transform=NormalizeFeatures())


# 加载CiteSeer数据集
dataset = Planetoid(root='./dataset', name="CiteSeer", transform=NormalizeFeatures())


# 加载Flickr数据集
# dataset = Flickr(root='./dataset/Flickr')   


# 选择其中一个数据集作为示例
data = dataset[0]  # 例如，使用Cora数据集




epochs = 11
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
# model = GAT(dataset.num_node_features, dataset.num_classes).to(device)
model = GraphSAGE(dataset.num_node_features, dataset.num_classes).to(device)
# model = GIN(dataset.num_node_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 全图训练

start_time = time.time()
for epoch in range(1, epochs):
    loss = train(model, data, optimizer, device)
    if epoch % 10 == 0:
        accuracy = test(model, data, device)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
full_graph_training_time = time.time() - start_time

print(f'Full graph training time: {full_graph_training_time:.4f} seconds')






# 使用NeighborSampler进行子图采样
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[15, 10], batch_size=128, shuffle=True, num_nodes=data.num_nodes)

def train_with_sampling(model, data, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs[0].edge_index)
        loss = F.nll_loss(out[:batch_size], data.y[n_id[:batch_size]].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
    return total_loss / len(train_loader.dataset)

# 子图采样训练
start_time = time.time()
for epoch in range(1, epochs):
    loss = train_with_sampling(model, data, train_loader, optimizer, device)
    if epoch % 10 == 0:
        accuracy = test(model, data, device)
        print(f'(Subgraph Sampling) Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
subgraph_sampling_training_time = time.time() - start_time

print(f'Subgraph sampling training time: {subgraph_sampling_training_time:.4f} seconds')