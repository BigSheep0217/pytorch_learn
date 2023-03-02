import torch.nn as nn
import torch


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.pad = nn.ConstantPad2d((3, 0, 3, 0), 0) # (左,右,上,下), 值
        self.net0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, stride=4)
        self.net1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=(1, 0, 1, 0), stride=4)
        self.net2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=(2), stride=4)
        self.net3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=3, stride=4)


    def forward(self, x):
        return self.net0(x), self.net1(x), self.net2(x), self.net3(x)
    

if __name__ == "__main__":
    model = ToyModel()
    data = torch.ones((1, 3, 1920, 1080))
    print(data.shape)
    output0, output1, output2, output3 = model(data)
    print(output0.shape)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)

