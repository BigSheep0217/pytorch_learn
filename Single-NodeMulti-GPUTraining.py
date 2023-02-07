import torch.nn as nn
import torch
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torchvision.models import resnet18
from torchvision.datasets import MNIST

import os
import argparse

def cleanup():
    dist.destroy_process_group()

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="local device id on current node")
    parser.add_argument("--world_size", type=int, default=2, help="world size")
    args = parser.parse_args()

    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=args.local_rank, world_size=args.world_size)
    model = ToyModel().to(args.local_rank)
    ddp_model = DDP(model, device_ids=[args.local_rank])
    
    dataset = MNIST()
    train_sampler = DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, sampler=train_sampler)


    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    
    
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    
    data = torch.randn(10, 10).to(args.local_rank)
    labels = torch.randn(10, 5).to(args.local_rank)
    
    outputs = ddp_model(data)
    
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # train_sampler.set_epoch(0)

    
    dist.destroy_process_group()
    
