import torch
import torch.nn as nn
from math import pi
import torch.nn.functional as F
import pyramid

import torchvision.models.resnet as res

def _resnet(arch, block, layers, pretrained, progress, **kwargs): #
    model = ReSpaceNet(block, layers, **kwargs)
    if pretrained:
        state_dict = res.load_state_dict_from_url(res.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class ReSpaceNet(res.ResNet):
    def __init__(self, block, layers,**kwargs):
        super().__init__(block,layers,**kwargs)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def patch(self):
        cw = self.conv1.weight
        self.conv1.stride = (1,1)
        self.conv1.weight = nn.Parameter(cw[:,[1],:,:].cuda())
        self.layer4 = None
        self.layer3[5].conv2.bias = nn.Parameter(torch.zeros(256))
        self.layer3[5].bn2 = nn.Sequential()
        self.fc = None

def mass2d(x):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    p = F.softmax(x.view(N,C,H*W), 2).view(N,C,H,W,1)
    # -7/8
    h_range = torch.arange((H - 1) / H, -1, -2 / H, device=x.device)
    w_range = torch.arange(-(W - 1) / W, 1, 2 / W, device=x.device)
    h_grid,w_grid = torch.meshgrid(h_range,w_range)
    grid = torch.stack((h_grid, w_grid), 2).expand(N, C, H, W, 2)
    mass = grid*p  # 24 256 8 8 1/2
    mass = torch.cat((p*x.unsqueeze(4),mass),dim=4) # 24 256 8 8 3
    com = mass.sum(dim=(2,3))
    return com, p.detach()

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(GraphConv, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=in_features, out_features=out_features)
        self.A = nn.Parameter((torch.ones(19,19)*(1/19)).float().cuda(), requires_grad=True)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.drop2 = nn.Dropout(p=0.2)
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.activation = activation


    def forward(self, X):
        batch = X.size(0)
        A_hat = self.A.unsqueeze(0).repeat(batch, 1, 1)

        X =  self.fc1(torch.bmm(A_hat, X))+ self.fc2(X)
        if self.activation is not None:
            X = self.drop1(self.activation(X))  #
        return X
class PyramidAttention(nn.Module):
    def __init__(self,levels):

        super().__init__()
        self.levels = levels

        self.resnet = _resnet('resnet34', res.BasicBlock,
                              [3, 4, 6, 3],
                              True, True)
        self.resnet.patch()
        self.graphConv1 = GraphConv(self.levels*768, 512)
        self.graphConv2 = GraphConv(512, 128)
        self.graphConv3 = GraphConv(128, 2, activation=None)

    def forward(self, x,poses,train=False):
        # pos N multi 19 2
        N, multi = poses.shape[0],poses.shape[1]
        landmarks = poses.shape[2]
        device = x[0].device
        resNetInput = torch.zeros(N, multi, landmarks, self.levels, 64, 64, device = poses.device)  # 2 2 6 64 64
        Rs = torch.zeros(N, multi, landmarks, 2,2, device = poses.device)
        scales = torch.zeros(N, multi, landmarks, 1,1, device = poses.device)
        for landmark in range(landmarks):
            pos = poses[:,:,landmark,:] #2 2 1 2
            theta = (torch.rand((pos.shape[0], pos.shape[1], 1, 1), device=device) * 2 - 1) * pi / 12  # 15.
            scale = torch.exp((torch.rand((pos.shape[0], pos.shape[1], 1, 1), device=device) * 2 - 1)*0.05)  #5%
            if not train:
                theta = theta*0
                scale = torch.ones((pos.shape[0], pos.shape[1], 1, 1), device=device)
            scales[:,:,landmark, :,:] = scale
            rsin = theta.sin()
            rcos = theta.cos()
            H = x[0].shape[2]
            W = x[0].shape[3]
            pos_fix = pos.clone()
            pos_fix[:, :, 1] = pos[:, :, 1] * (W / H)   #
            R = torch.cat((rcos, -rsin, rsin, rcos), 3).view(pos.shape[0], pos.shape[1], 2, 2)
            Rs[:, :, landmark, :, :] = R
            # T 2,2,2,3
            T = torch.cat((R*scale, pos_fix.unsqueeze(3)), 3)
            s = 64
            stacked = pyramid.stack(x, s, T, augment=train)
            resNetInput[:,:,landmark,:,:,:] = stacked.detach()

        batched = resNetInput.view(N*multi*landmarks*self.levels,1,s,s) # 2x2x19x6
        out = self.resnet(batched)
        out, self.heat_vis = mass2d(out)
        self.mass_vis = out.detach() # 456 256 3  ！！！！ 4*6*19   = 456

        out = out.view(N*multi, landmarks, -1)
        out = self.graphConv1(out)
        out = self.graphConv2(out)
        out = self.graphConv3(out)
        out = out.view(N,multi,landmarks,1,2)
        out = torch.matmul(out,Rs.transpose(3,4)/scales)  # inverse！！！

        return out.view(N,multi,landmarks,2)

def load_model(levels,name,load=False):
    model = PyramidAttention(levels)
    if load:
        model.load_state_dict(torch.load(f"runs/{name}.pth"))
        print('load successfully, #params {:1.4}m'.format(sum([param.nelement() for param in model.parameters()])/1000000) )

    return model


