import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(20, 1)

    def forward(self, x):
        return self.net1(x)
    

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
       

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='start single Node with SingleGpu')
    args = parser.parse_args()
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    
    model = ToyModel()

    dataset = MyTrainDataset(128)
    train_loader = DataLoader(dataset, batch_size=1)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    model.to(device)
    
    for epoch in range(10):
        b_sz = len(next(iter(train_loader))[0])
        print(f"[GPU{device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}")
        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    

    
    