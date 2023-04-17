import torch


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(20, 1)

    def forward(self, x):
        return self.net1(x)
    
# model = ToyModel()

# torch.save(model, 'best.pt')

model = torch.load('best.pt')