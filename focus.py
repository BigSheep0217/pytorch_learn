import torch.nn as nn
import torch
from PIL import Image
import torchvision.transforms as T


if __name__ == "__main__":
    img = Image.open('1.jpg')
    trans = T.ToTensor()
    trans_back = T.ToPILImage()
    x = trans(img).unsqueeze(0) # ([1, 3, 720, 1280])
    
    b, c, h, w = x.size()
    s = 2
    y = x.view(b, c, h // s, s, w // s, s) # ([1, 3, 360, 2, 640, 2])
    
    z = y.permute(0, 3, 5, 1, 2, 4).contiguous() # ([1, 2, 2, 3, 360, 640])

    # u = x.view(b, c * s * s, h // s, w // s)
    u = z.view(b, c * s * s, h // s, w // s)
    print(u.shape)
    back_img = trans_back(u[0,6:9, ...])
    back_img.show()
    
    # u = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
    # print(u.shape)
    # back_img = trans_back(u[0,0, ...])
    # back_img.show()
    
    # u = torch.cat((z[:, 0, 0, :, ...], z[:, 1, 0, :, ...], z[:, 0, 1, :, ...], z[:, 1, 1, :, ...]), 1)
    # print(u.shape)
    # back_img = trans_back(u[0, 0:3, ...])
    # back_img.show()