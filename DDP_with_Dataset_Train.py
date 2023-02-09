import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    

def main():
    # 1-初始化线程
    dist.init_process_group(backend="nccl")
    
    dataset, model, optimizer = MyTrainDataset(128), torch.nn.Linear(20, 1), torch.optim.SGD(model.parameters(), lr=1e-3)
    
    train_data = DataLoader(dataset,batch_size=1,pin_memory=True,shuffle=False,sampler=DistributedSampler(dataset))
    
    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    ddp_model = DDP(model, device_ids=[gpu_id])
    
    for epoch in range(10):
        b_sz = len(next(iter(train_data))[0])
        print(f"[GPU{gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_data)}")
        train_data.sampler.set_epoch(epoch)
        for source, targets in train_data:
            source = source.to(gpu_id)
            targets = targets.to(gpu_id)
            optimizer.zero_grad()
            output = ddp_model(source)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
        
    dist.destroy_process_group()

if __name__ == '__main__':

    main()
