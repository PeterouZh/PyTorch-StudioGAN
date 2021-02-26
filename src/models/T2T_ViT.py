import math
import numpy as np
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from utils.model_ops import *
from utils.misc import *

import torch
import torch.nn as nn
import torch.nn.functional as F



class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, z_dims_after_concat):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if self.conditional_bn:
            self.bn1 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=in_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
            self.bn2 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=out_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
        else:
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, label):
        x0 = x
        if self.conditional_bn:
            x = self.bn1(x, label)
        else:
            x = self.bn1(x)

        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.conv2d1(x)
        if self.conditional_bn:
            x = self.bn2(x, label)
        else:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, shared_dim, img_size, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block, activation_fn,
                 conditional_strategy, num_classes, initialize, G_depth, mixed_precision):
        super(Generator, self).__init__()
        g_in_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                "64": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "128": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "256": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "512": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim]}

        g_out_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                 "64": [g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "128": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "256": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "512": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim, g_conv_dim]}
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN", "ContraGAN++"] else False

        self.in_dims =  g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.n_blocks = len(self.in_dims)
        self.chunk_size = z_dim//(self.n_blocks+1)
        self.z_dims_after_concat = self.chunk_size + self.shared_dim
        assert self.z_dim % (self.n_blocks+1) == 0, "z_dim should be divided by the number of blocks "

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom)
        else:
            self.linear0 = linear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom)

        self.shared = embedding(self.num_classes, self.shared_dim)

        self.blocks = []
        for index in range(self.n_blocks):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                      out_channels=self.out_dims[index],
                                      g_spectral_norm=g_spectral_norm,
                                      activation_fn=activation_fn,
                                      conditional_bn=conditional_bn,
                                      z_dims_after_concat=self.z_dims_after_concat)]]

            if index+1 == attention_after_nth_gen_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = batchnorm_2d(in_features=self.out_dims[-1])

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d5 = snconv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d5 = conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)


    def forward(self, z, label, shared_label=None, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            zs = torch.split(z, self.chunk_size, 1)
            z = zs[0]
            if shared_label is None:
                shared_label = self.shared(label)
            else:
                pass
            labels = [torch.cat([shared_label, item], 1) for item in zs[1:]]

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            counter = 0
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, Self_Attn):
                        act = block(act)
                    else:
                        act = block(act, labels[counter])
                        counter +=1

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out


class Attention_T2T(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sn=False):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.activation = nn.ReLU(inplace=True)

        if sn:
            self.qkv = snlinear(dim, in_dim * 3, bias=qkv_bias)
            self.proj = snlinear(in_dim, in_dim)
        else:
            self.qkv = linear(dim, in_dim * 3, bias=qkv_bias)
            self.proj = linear(in_dim, in_dim)

        self.attn_drop = dropout(attn_drop)
        self.proj_drop = dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.activation((q @ k.transpose(-2, -1)) * self.scale) + self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection
        return x


class Attention_backbone(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.activation = nn.ReLU(inplace=True)

        if sn:
            self.qkv = snlinear(dim, dim * 3, bias=qkv_bias)
            self.proj = snlinear(dim, dim)
        else:
            self.qkv = linear(dim, dim * 3, bias=qkv_bias)
            self.proj = linear(dim, dim)

        self.attn_drop = dropout(attn_drop)
        self.proj_drop = dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.activation((q @ k.transpose(-2, -1)) * self.scale) + self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=nn.ReLU(), drop=0., sn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if sn:
            self.fc1 = snlinear(in_features, hidden_features)
            self.fc2 = snlinear(hidden_features, out_features)
        else:
            self.fc1 = linear(in_features, hidden_features)
            self.fc2 = linear(hidden_features, out_features)

        self.act = activation
        self.drop = dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Token_transformer(nn.Module):
    def __init__(self, dim, in_dim, batch_size, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., activation=nn.ReLU(), sn=False):
        super().__init__()
        self.batch_size = batch_size
        self.sn = sn
        self.norm1 = Layernorm2d(self.batch_size, dim, self.sn)

        self.attn = Attention_T2T(dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, sn=self.sn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Layernorm2d(self.batch_size, in_dim, self.sn)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, activation=activation, drop=drop, sn=self.sn)


    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block(nn.Module):
    def __init__(self, dim, batch_size, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., activation=nn.ReLU(), sn=False):
        super().__init__()
        self.batch_size = batch_size
        self.sn = sn
        self.norm1 = Layernorm2d(self.batch_size, dim, self.sn)

        self.attn = Attention_backbone(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=drop, sn=self.sn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = Layernorm2d(self.batch_size, dim, self.sn)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation, drop=drop, sn=self.sn)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, dataset, img_size, batch_size, tokens_type, in_chans, embed_dim, token_dim, activation, sn):
        super().__init__()
        f_kernel = 3 if dataset == 'CIFAR10' else 7
        f_stride = 2 if dataset == 'CIFAR10' else 4
        f_padding = 1 if dataset == 'CIFAR10' else 2
        self.batch_size = batch_size
        self.sn = sn

        self.soft_split0 = nn.Unfold(kernel_size=(f_kernel, f_kernel), stride=(f_stride, f_stride), padding=(f_padding, f_padding))
        self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.attention1 = Token_transformer(dim=in_chans * f_kernel * f_kernel, in_dim=token_dim, batch_size=self.batch_size,
                                            num_heads=1, mlp_ratio=1.0, activation=activation, sn=self.sn)
        self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, batch_size=self.batch_size,
                                            num_heads=1, mlp_ratio=1.0, activation=activation, sn=self.sn)
        if self.sn:
            self.project = snlinear(token_dim * 3 * 3, embed_dim)
        else:
            self.project = linear(token_dim * 3 * 3, embed_dim)

        self.num_patches = (img_size // (f_stride * 2 * 2)) * (img_size // (f_stride * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately


    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: restricturization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: restricturization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dataset, img_size, batch_size, tokens_type, token_dim, num_classes, embed_dim, depth, num_heads, mlp_ratio,
                 hypersphere_dim, bottleneck_dim, normalize_embed, qkv_bias, qk_scale, activation_fn, conditional_strategy, drop_rate,
                 attn_drop_rate, drop_path_rate, d_spectral_norm, initialize, mixed_precision):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.normalize_embed = normalize_embed
        self.conditional_strategy = conditional_strategy
        self.d_spectral_norm = d_spectral_norm
        self.mixed_precision = mixed_precision

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.tokens_to_token = T2T_module(dataset=self.dataset, img_size=img_size, batch_size=batch_size, tokens_type=tokens_type, in_chans=3,
                                          embed_dim=embed_dim, token_dim=token_dim, activation=self.activation, sn=self.d_spectral_norm)
        num_patches = self.tokens_to_token.num_patches

        n_token = 1
        self.adv_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.conditional_strategy != 'no':
            n_token += 1
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + n_token, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, batch_size=self.batch_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], activation=self.activation, sn=self.d_spectral_norm)
            for i in range(depth)])

        self.norm = Layernorm2d(self.batch_size, embed_dim, self.d_spectral_norm)

        # Classifier head
        if d_spectral_norm:
            self.adv_head = snlinear(in_features=embed_dim, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
                self.proj_head0 = snlinear(in_features=embed_dim, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.proj_head1 = snlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
            elif self.conditional_strategy == 'ContraGAN++':
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
                self.proj_head = snlinear(in_features=embed_dim, out_features=hypersphere_dim)
                self.convert_head0 = snlinear(in_features=hypersphere_dim, out_features=bottleneck_dim)
                self.convert_head1 = snlinear(in_features=bottleneck_dim, out_features=hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = sn_embedding(num_classes, embed_dim)
            elif self.conditional_strategy == 'ACGAN':
                self.cls_head = snlinear(in_features=embed_dim, out_features=num_classes)
            else:
                pass
        else:
            self.adv_head = linear(in_features=embed_dim, out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.embedding = embedding(num_classes, hypersphere_dim)
                self.proj_head0 = linear(in_features=embed_dim, out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.proj_head1 = linear(in_features=hypersphere_dim, out_features=hypersphere_dim)
            elif self.conditional_strategy == 'ContraGAN++':
                self.embedding = embedding(num_classes, hypersphere_dim)
                self.proj_head = linear(in_features=embed_dim, out_features=hypersphere_dim)
                self.convert_head0 = linear(in_features=hypersphere_dim, out_features=bottleneck_dim)
                self.convert_head1 = linear(in_features=bottleneck_dim, out_features=hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = embedding(num_classes, embed_dim)
            elif self.conditional_strategy == 'ACGAN':
                self.cls_head = linear(in_features=embed_dim, out_features=num_classes)
            else:
                pass

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

        trunc_normal_(self.adv_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        adv_tokens = self.adv_token.expand(B, -1, -1)
        if self.conditional_strategy == 'no':
            x = torch.cat((adv_tokens, x), dim=1)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((adv_tokens, cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:,0,:], x[:,1,:]


    def forward(self, x, label, evaluation=False, fake=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            if self.conditional_strategy == 'no':
                adv_features, _ = self.forward_features(x)
                authen_output = torch.squeeze(self.adv_head(adv_features))
                return authen_output

            elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                adv_features, cls_features = self.forward_features(x)
                cls_proxy = self.embedding(label)
                cls_embed = self.proj_head0(cls_features)
                authen_output = torch.squeeze(self.adv_head(adv_features))
                if self.nonlinear_embed:
                    cls_embed = self.proj_head1(self.activation(cls_embed))
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ContraGAN++':
                adv_features, cls_features = self.forward_features(x)
                cls_proxy = self.embedding(label)
                cls_embed = self.proj_head(cls_features)
                authen_output = torch.squeeze(self.adv_head(adv_features))
                if fake:
                    cls_embed = self.convert_head0(self.activation(cls_embed))
                    cls_embed = self.convert_head1(self.activation(cls_embed))
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ProjGAN':
                adv_features, cls_features = self.forward_features(x)
                proj = torch.sum(torch.mul(self.embedding(label), cls_features), 1)
                authen_output = torch.squeeze(self.adv_head(adv_features))
                return proj + authen_output

            elif self.conditional_strategy == 'ACGAN':
                adv_features, cls_features = self.forward_features(x)
                cls_output = self.cls_head(cls_features)
                authen_output = torch.squeeze(self.adv_head(adv_features))
                return cls_output, authen_output

            else:
                raise NotImplementedError