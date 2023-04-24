import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel as DDP
import argparse
import os
import time


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(20, 1)

    def forward(self, x):
        return self.net1(x)
    
def demo_basic(local_rank):

    # local_world_size 实际上应该是 dist.get_world_size()获取的
    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and rank 1 uses GPUs [4, 5, 6, 7].
    # n = torch.cuda.device_count() // local_world_size # 原资料local_world_size这个参数容易引起误解
    n = torch.cuda.device_count() // dist.get_world_size() # 自动计算每个节点的每个进程用多少卡
    device_ids = list(range(local_rank * n, (local_rank + 1) * n)) # 计算第local_rank个进程所需要用到的GPU

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, " # 这里是进程的rank，表示第几个，一共有nproc_per_node个，由launch.py自动分配
        + f"world_size = {dist.get_world_size()}, " # 一共有多少个进程，来自参数--nproc_per_node=2
        + f"n = {n}, " # 
        + f"device_ids = {device_ids}"
    )

    model = ToyModel().cuda(device_ids[0])
    
    ddp_model = DDP(model, device_ids)
    
    outputs = ddp_model(torch.randn(10, 20))
    print(f"demo done !!! outputs: {outputs}")
    # time.sleep(1)
    
def spmd_main(local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    
    dist.init_process_group(backend="nccl")
    
    # print(
    #     f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
    #     + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    # )

    demo_basic(local_rank)

    # Tear down the process group
    dist.destroy_process_group()
    
# python /opt/conda/envs/torch110/lib/python3.7/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=2 launch_DDP.py
# --nnode=1 --node_rank=0 --nproc_per_node=2 launch_DDP.py
# 不推荐用python launch.py启动
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0) # launch.py 给每个进程的排序值
    # parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    # local_world_size=1 说明每个节点只用一个进程，launch.py 给这个进程分配的rank是0, local_rank=0 
    # spmd_main(args.local_world_size, args.local_rank)
    
    spmd_main(args.local_rank)