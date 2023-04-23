from logging import Logger
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

def load_snapshot(model, snapshot_path):
    if os.path.isfile(snapshot_path) and snapshot_path.endwith(".pt"):
        print("Loading checkpoint: {}...".format(snapshot_path))
        snapshot = torch.load(snapshot_path)
        model.load_state_dict(snapshot["MODEL_STATE"])
        epochs_run = snapshot["EPOCHS_RUN"]
    else:
        print("!!! not find checkpoint: {}...".format(snapshot_path))
        epochs_run = 0
    return model, epochs_run

def save_snapshot(model, epoch, snapshot_path):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "EPOCHS_RUN": epoch,
    }
    # torch.save(snapshot, "snapshot_path/")
    print(f"Epoch {epoch} | Training snapshot saved at snapshot_path")
    

def prepare_DDPmodel(gpu_id, snapshot_path):
    model = torch.nn.Linear(20, 1)
    model = model.to(gpu_id) # 先送入GPU
    
    model, epochs_run = load_snapshot(model, snapshot_path) # 模型载入
    
    ddp_model = DDP(model, device_ids=[gpu_id]) # 最后包裹
    
    return ddp_model, epochs_run

def prepare_DDPsampler(dataset):
    DDP_sampler = DistributedSampler(dataset)
    return DDP_sampler

def prepare_dataloader(dataset, sampler):
    data_loader = DataLoader(dataset,batch_size=1,pin_memory=True,shuffle=False,sampler=sampler)
    return data_loader

def prepare_dataset():
    dataset = MyTrainDataset(128)
    return dataset

def prepare_optm(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return optimizer

def prepare_loss():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn

def run_batch(source, targets, optimizer, model, loss_fn, gpu_id):
    source = source.to(gpu_id)
    targets = targets.to(gpu_id)
    optimizer.zero_grad()
    output = model(source)
    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()

def save_condition(epoch):
    return False

# torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py (args1 args2 ...)

def main():
    # 1-初始化线程
    dist.init_process_group(backend="nccl")
    
    gpu_id = int(os.environ["LOCAL_RANK"])
    
    dataset = prepare_dataset()
    
    model, epochs_run = prepare_DDPmodel(gpu_id, "snapshot_path")
    
    optimizer = prepare_optm(model)
    
    DDP_sampler = prepare_DDPsampler(dataset)
    
    train_data = prepare_dataloader(dataset, DDP_sampler)
    
    loss_fn = prepare_loss()
    
    for epoch in range(epochs_run, 10):
        b_sz = len(next(iter(train_data))[0])
        print(f"[GPU{gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_data)}")
        train_data.sampler.set_epoch(epoch)
        for source, targets in train_data:
            run_batch(source, targets, optimizer, model, loss_fn, gpu_id)
        
        if gpu_id==0 and save_condition(epoch):
            save_snapshot(model, epoch, "snapshot_path")
            
    dist.destroy_process_group()

if __name__ == '__main__':

    main()
