import os
import sys
import time
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
# import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()
    
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
    
def demo_basic(world_size):
    
    setup()
    
    rank = dist.get_rank()
    # 自动分配GPU资源，因为model需要2张GPU
    dev0 = (rank * 2) % world_size # 0 2 4 6
    dev1 = (rank * 2 + 1) % world_size # 1 3 5 7
    print(f"Running basic DDP example on rank {rank} , use GPU [{dev0} {dev1}] / {world_size}.")
    
    # create model and move it to GPU with id rank
    model = ToyMpModel(dev0, dev1) # （0,1）在别的rank进程中为（2,3）（4,5）（6,7），也就是8卡4进程
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss = loss_fn(outputs, labels)
    # optimizer.step()
    print(f"demo done !!! loss: {loss}")
    cleanup()

# torchrun --standalone --nnodes=1 --nproc_per_node=2 Model_DDP.py

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    demo_basic(n_gpus)
    