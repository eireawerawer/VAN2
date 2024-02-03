import torch
import torch.nn as nn
x = torch.ones(1, 32, 64, 64)
class AMlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., change=0, bias=False, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x.permute(0, 3, 1, 2)

module = AMlp(32)
x = module(x)
print(x.shape)