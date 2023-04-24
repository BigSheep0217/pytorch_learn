"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 在单机多卡上实现通信

def run(rank, size):
    # """ Distributed function to be implemented later. """
    # # isend() and irecv() return distributed request objects when used. In general, the type of this object is unspecified as they should never be created manually, but they are guaranteed to support two methods:
    # # is_completed() - returns True if the operation has finished
    # # wait() - will block the process until the operation is finished. is_completed() is guaranteed to return True once it returns.
    # tensor = torch.zeros(1) # 请注意，进程 1 需要分配内存以存储它将接收的数据。
    # req = None
    
    # if rank == 0:
    #     tensor += 1
    #     # Send the tensor to process 1
    #     req = dist.isend(tensor=tensor, dst=1)
    #     # print('Rank 0 started sending')
    #     print(f"rank {rank} sending")
    # else:
    #     # Receive tensor from process 0
    #     req = dist.irecv(tensor=tensor, src=0)
    #     # print('Rank 1 started receiving')
    #     print(f"rank {rank} receiving")
    # req.wait() # 确保发送或接收完成
    # print('Rank ', rank, ' has data ', tensor[0])
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()