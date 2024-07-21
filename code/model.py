import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm2d(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-6) * self.weight + self.bias
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SCA(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Conv2d(nf, nf, 3, padding=1)

    def forward(self, x):
        return x * self.body(x)

class NAFBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(nf),
            nn.Conv2d(nf, 2*nf, 1),
            nn.Conv2d(2*nf, 2*nf, 3, padding=1, groups=2*nf),
            SimpleGate(),
            nn.Conv2d(nf, nf, 3, padding=1),
            SCA(nf),
        )

    def forward(self, x):
        return x + self.body(x)

class Stage(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor([1.0]))
        self.body = nn.Sequential(*[NAFBlock(nf) for _ in range(2)])
    
    def forward(self, x):
        x, y, A, AT = x
        z = F.pixel_shuffle(x[:, :4], 2)
        z = z - self.rho * AT(A(z) - y.to(z.device))
        x = self.body(torch.cat([F.pixel_unshuffle(z, 2), x[:, 4:]], dim=1))
        return x, y, A, AT

class Net(nn.Module):
    def __init__(self, nb, A32, A128, nf):
        super(Net, self).__init__()
        self.A32 = nn.Parameter(A32, requires_grad=False)
        self.alpha32 = nn.Parameter(torch.Tensor([1e-2 for _ in range(512)]).view(1, 512))
        self.beta32 = nn.Parameter(torch.zeros(1, 512))
        self.gamma32 = nn.Parameter(torch.Tensor([1e-2 for _ in range(512)]).view(1, 512))
        self.A128 = nn.Parameter(A128, requires_grad=False)
        self.alpha128 = nn.Parameter(torch.Tensor([1e-2 for _ in range(5000)]).view(1, 5000))
        self.beta128 = nn.Parameter(torch.zeros(1, 5000))
        self.gamma128 = nn.Parameter(torch.Tensor([1e-2 for _ in range(5000)]).view(1, 5000))
        self.head = nn.Sequential(nn.Conv2d(2, nf//4, 3, padding=1), nn.PixelUnshuffle(2))
        self.body = nn.Sequential(*[Stage(nf) for _ in range(nb)])
        self.tail = nn.Sequential(nn.Conv2d(nf, 4, 3, padding=1), nn.PixelShuffle(2))
 
    def forward(self, y_input, q, sr):
        b = y_input.shape[0]
        q = q[0,:b].unsqueeze(1).expand(b,1)
        if sr == 4:
            max_q = 512
            perm = torch.randperm(max_q, device=q.device)
            mask = (torch.arange(max_q,device=q.device).view(1,max_q).expand(b,max_q) < q)
            y = (y_input * self.alpha32 + self.beta32)[:, perm] * mask
            A = (self.A32 * self.gamma32)[:, perm]
            AT = A.t()
            A_func = lambda z: F.avg_pool2d(z, 4, stride=4).view(b, -1).mm(A) * mask
            AT_func = lambda z: F.interpolate(z.mm(AT).view(-1, 1, 32, 32), scale_factor=4)
        elif sr == 1:
            max_q = 5000
            perm = torch.randperm(max_q, device=q.device)
            mask = (torch.arange(max_q,device=q.device).view(1,max_q).expand(b,max_q) < q)
            y = (y_input * self.alpha128 + self.beta128)[:, perm] * mask
            A = (self.A128 * self.gamma128)[:, perm]
            AT = A.t()
            A_func = lambda z: z.view(b, -1).mm(A) * mask
            AT_func = lambda z: z.mm(AT).view(-1, 1, 128, 128)
        x_init = AT_func(y)
        cs_ratio_map = (q / 16384).view(b,1,1,1).expand_as(x_init)
        x = torch.cat([x_init, cs_ratio_map], dim=1)
        x = self.head(x)
        x = self.body([x, y, A_func, AT_func])[0]
        x = self.tail(x)
        return x.reshape(b, 16384)
    
if __name__ == "__main__":
    model = Net(9, torch.rand(1024,512), torch.rand(16384, 5000), 128).cuda()
    x = torch.rand(16, 16384).cuda()
    y = torch.rand(16, 5000).cuda()
    q = torch.randint(low=1,high=5001,size=(16,1)).cuda()
    x_out = model(y, q, 1)
    print(y.shape)
    print(x_out.shape)
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt/1e6, "M")
