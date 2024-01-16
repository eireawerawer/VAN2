import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return (self.scale.unsqueeze(-1).unsqueeze(-1) * self.relu(x)**2) + self.bias.unsqueeze(-1).unsqueeze(-1)
   
class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., mlp=0):
        super().__init__()
        self.mlp = mlp
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv1 = DWConv(hidden_features)
        if mlp == 1:
            # self.act1 = act_layer()
            self.dwconv2 = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv1(x)
        if self.mlp == 1:
            # x = self.act1(x)
            x = self.dwconv2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AMlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., mlp=0, bias=False, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x.permute(0, 3, 1, 2)

class LKA(nn.Module):
    def __init__(self, in_c, act=nn.GELU, nopw=0):
        super().__init__()
        self.nopw = nopw
        if nopw == 0:
            self.proj_1 = nn.Conv2d(in_c, in_c, 1)
            self.proj_2 = nn.Conv2d(in_c, in_c, 1)

        self.activation = act()
        self.conv0 = nn.Conv2d(in_c, in_c, 5, padding=2, groups=in_c)
        self.conv_spatial = nn.Conv2d(in_c, in_c, 7, stride=1, padding=9, groups=in_c, dilation=3)
        self.conv1 = nn.Conv2d(in_c, in_c, 1)

    def forward(self, x):
        shorcut = x.clone()
        if self.nopw==0:
            x = self.proj_1(x)
            x = self.activation(x)

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        x *= attn

        if self.nopw==0:
            x = self.proj_2(x)
        x += shorcut
        return x

class ILKA(nn.Module):
    def __init__(self, d_model, act=nn.GELU, nopw=0):
        super().__init__()
        self.nopw = nopw
        if nopw == 0:
            self.proj_1 = nn.Conv2d(d_model, d_model, 1)
            self.proj_2 = nn.Conv2d(d_model, d_model, 1)


        self.activation = act()
        self.conv0 = nn.Conv2d(d_model, d_model, 5, padding=2, groups=d_model)
        self.conv_spatial = nn.Conv2d(d_model, d_model, 7, stride=1, padding=9, groups=d_model, dilation=3)
        self.conv1 = nn.Conv2d(d_model, d_model, 1)



    def forward(self, x):
        shorcut = x.clone()
        if self.nopw==0:
            x = self.proj_1(x)
            x = self.activation(x)

        c_attn = self.conv1(x)     
        s_attn = self.conv0(x)
        s_attn = self.conv_spatial(s_attn)

        x *= c_attn * s_attn
    
        if self.nopw==0:
            x = self.proj_2(x)
        x = x + shorcut
        return x

class TLKA(nn.Module):
    def __init__(self, in_c, act=nn.GELU, nopw=0):
        super().__init__()
        self.nopw = nopw
        if nopw == 0:
            self.proj_1 = nn.Conv2d(in_c, in_c, 1)
            self.proj_2 = nn.Conv2d(in_c, in_c, 1)

        self.activation = act()
        self.conv0 = nn.Conv2d(in_c, in_c, 5, padding=2, groups=in_c)
        self.conv_spatial = nn.Conv2d(in_c, in_c, 7, stride=1, padding=9, groups=in_c, dilation=3)
        self.conv1 = nn.Conv2d(in_c, in_c, 1)

    def forward(self, x):
        shorcut = x.clone()
        if self.nopw==0:
            x = self.proj_1(x)
            x = self.activation(x)

        attn = self.conv1(x)
        attn = self.conv0(attn)
        attn = self.conv_spatial(attn)

        x *= attn

        if self.nopw==0:
            x = self.proj_2(x)
        x += shorcut
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, nopw=0, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)



# slka
# nopw
# resscale
# starrelu
# convffn+


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 tokenmixer=LKA, 
                 mlp=AMlp, 
                 nopw=False, 
                 scale=0
                 ):
        super().__init__()
        self.scale = scale
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = tokenmixer(dim, nopw=nopw)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = mlp(in_features=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop, mlp=mlp)
     
        layer_scale_init_value = 1e-2      
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        if scale == 1:
            self.res_scale1 = Scale(dim=dim)
            self.res_scale2 = Scale(dim=dim)


        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        if self.scale == 0:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))


        elif self.scale == 1:
            x = (self.res_scale1(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
            x = (self.res_scale2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x




class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """


    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)


        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class VAN2(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2],
                num_classes=1000, mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., 
                norm_layer=nn.LayerNorm, num_stages=4, mlp=[Mlp, Mlp, Mlp, Mlp], act=StarReLU, scale=1, tokenmixer=[ILKA, ILKA, ILKA, ILKA], nopw=1):
        super().__init__()

        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0


        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])


            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], mlp=mlp[i], \
                      tokenmixer=tokenmixer[i], act_layer=act, nopw=nopw, scale=scale)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]


            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()


        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward_features(self, x):
        B = x.shape[0]


        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = x.mean(dim=1)
        return x


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)


        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)


    def forward(self, x):
        x = self.dwconv(x)
        return x

@register_model
def van2_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ablation_van2_nonopw_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        nopw=0,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ablation_van2_ILKA_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        act=nn.GELU, scale=0, nopw=0,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ablation_van2_TLKA_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        act=nn.GELU, scale=0, nopw=0, tokenmixer=[TLKA, TLKA, TLKA, TLKA],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ablation_van2_nopw_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        act=nn.GELU, scale=0, tokenmixer=[LKA, LKA, LKA, LKA],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ablation_van2_scale_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        act=nn.GELU, nopw=0, tokenmixer=[LKA, LKA, LKA, LKA], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ablation_van2_act_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        scale=0, nopw=0, tokenmixer=[LKA, LKA, LKA, LKA], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

def van2a_b0(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2_b1(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2a_b1(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2_b2(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2a_b2(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2_b3(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2a_b3(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2_b4(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2a_b4(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def van2_b5(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[96, 192, 480, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 24, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2a_b5(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[96, 192, 480, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 24, 3],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def van2_b6(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[96, 192, 384, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[6,6,90,6],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def van2a_b6(pretrained=False, **kwargs):
    model = VAN2(
        embed_dims=[96, 192, 384, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[6,6,90,6],
        tokenmixer=[ILKA, ILKA, SelfAttention, SelfAttention], mlp=[Mlp, Mlp, AMlp, AMlp],
        **kwargs)
    model.default_cfg = _cfg()
    return model