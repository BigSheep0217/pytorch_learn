import torch.nn as nn
import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import argparse


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# 调用命令 节点数nnodes nproc_per_node每个节点GPU数
# torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py (args1 args2 ...)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='start single Node with MultiGpu by torchrun')
    args = parser.parse_args()
    # 0-使用torchrun不再需传入wordsize变量

    # 1-初始化线程
    dist.init_process_group(backend="nccl")
    
    # 2-LOCAL_RANK由torchrun自动设置
    gpu_id = int(os.environ["LOCAL_RANK"])
    
    # 3-os.environ['WORLD_SIZE']由torchrun自动设置
    print(f"gpu_id : {gpu_id}")
    print(f"!!! distributed WORLD_SIZE : {os.environ['WORLD_SIZE']}")
    print(f"!!! distributed LOCAL_RANK : {os.environ['LOCAL_RANK']}")
    print(f"!!! distributed RANK : {dist.get_rank()}")
    
    model = ToyModel().to(gpu_id)
    
    ddp_model = DDP(model, device_ids=[gpu_id])
    
    loss_fn = nn.MSELoss()
    
    data = torch.randn(10, 10).to(args.local_rank)
    labels = torch.randn(10, 5).to(args.local_rank)
    
    outputs = ddp_model(data)
    
    loss = loss_fn(outputs, labels)
    
    print(f"demo done !!! loss: {loss}")
    
    dist.destroy_process_group()
    
