
import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch import einsum
from collections.abc import Iterable
import os
import copy
from einops import rearrange


# MVCS Implementation
norm_func = nn.InstanceNorm3d

class SSA(nn.Module):
    '''
    dim: c
    n_segment: d   
    '''
    def __init__(self, dim, n_segment):
        super(SSA, self).__init__()
        self.scale = dim ** -0.5    
        self.n_segment = n_segment  

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size = 1)
        self.attend = nn.Softmax(dim = -1)

        self.to_temporal_qk = nn.Conv3d(dim, dim * 2, 
                                  kernel_size=(3, 1, 1), 
                                  padding=(1, 0, 0))

    def forward(self, x):

        bt, c, h, w = x.shape
        t = self.n_segment
        b = bt / t

        # Spatial Attention:
        qkv = self.to_qkv(x) # bt, 3*c, h, w
        q, k, v = qkv.chunk(3, dim = 1) # bt, c, h, w
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v)) # bt, hw, c

        # -pixel attention
        pixel_dots = einsum('b i c, b j c -> b i j', q, k) * self.scale # bt, hw, hw
        pixel_attn = torch.softmax(pixel_dots, dim=-1) # bt, hw, hw
        pixel_out = einsum('b i j, b j d -> b i d', pixel_attn, v) # bt, hw, c
        
        # -channel attention
        chan_dots = einsum('b i c, b i k -> b c k', q, k) * self.scale # bt, c, c
        chan_attn = torch.softmax(chan_dots, dim=-1)
        chan_out = einsum('b i j, b d j -> b d i', chan_attn, v) # bt, hw, c
        
        # aggregation
        x_hat = pixel_out + chan_out # bt, hw, c
        x_hat = rearrange(x_hat, '(b t) (h w) c -> b c t h w', t=t, h=h, w=w) # b c t h w  
        
        # Temporal attention
        t_qk = self.to_temporal_qk(x_hat) # b, 2*c, t, h, w
        tq, tk = t_qk.chunk(2, dim=1) # b, c, t, h, w
        tq, tk = map(lambda t: rearrange(t, 'b c t h w -> b t (c h w )'), (tq, tk)) # b, t, chw
        tv = rearrange(v, '(b t) (h w) c -> b t (c h w)', t=t, h=h, w=w) # shared value embedding ; b, t, chw
        dots = einsum('b i d, b j d -> b i j', tq, tk) # b, t, t
        attn = torch.softmax(dots, dim=-1)
        out = einsum('b k t, b t d -> b k d', attn, tv) # b, t, chw
        out = rearrange(out, 'b t (c h w) -> (b t) c h w', h=h,w=w,c=c) # bt, c, h, w

        # the same size between input and output
        return out


# 定义SADA_Attention类，用于实现SADA注意力模型
class SADA_Attention(nn.Module):
    def __init__(self, inchannel, n_segment,dim=256,heads=4,dropout=0.2):
        super(SADA_Attention, self).__init__()

        # 定义LF0，LF1，LF2
        self.LF0 = SSA(inchannel, n_segment)
        self.LF1 = SSA(inchannel, n_segment)
        self.LF2 = SSA(inchannel, n_segment)  

        # multi-view attention
        # 定义多视图注意力
        self.self_mattn_01 = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_mattn_02 = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_mattn_10 = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_mattn_12 = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_mattn_20 = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_mattn_21 = MultiheadAttention(dim, heads,dropout=dropout)

    def forward(self, x):

        n, c, d, w, h = x.size()

        # 将输入x的维度转换为n*d, c, w, h
        localx = copy.copy(x).transpose(1,2).contiguous().view(n*d, c, w, h)  # n*d, c, w, h
        # 计算LF0
        localx = self.LF0(localx).transpose(1,2).contiguous() # n*d, w, c, h
        # 将计算结果转换为n, c, d, w, h
        x0 = localx.view(n, c, d, w, h)  # n, c, d, w, h 

        # 将输入x的维度转换为n*w, c, d, h
        localx = copy.copy(x).permute(0, 3, 1, 2, 4).contiguous().view(n*w, c, d, h)  # n*w, c, d, h
        # 计算LF1
        localx = self.LF1(localx) # n*w, c, d, h
        # 将计算结果转换为n, w, c, d, h
        x1 = localx.view(n, w, c, d, h).permute(0, 2, 3, 1, 4).contiguous()  # n, c, d, w, h 

        # 将输入x的维度转换为n*h, c, d, w
        localx = copy.copy(x).permute(0, 4, 1, 2, 3).contiguous().view(n*h, c, d, w)  # n*h, c, d, w
        # 计算LF2
        localx = self.LF2(localx) # n*h, c, d, w
        # 将计算结果转换为n, h, c, d, w
        x2 = localx.view(n, h, c, d, w).permute(0, 2, 3, 4, 1).contiguous()  # n, c, d, w, h 

        # multi-view attention
        # 计算多视图注意力
        x0 = self.self_mattn_01(x0,x1,x1)[0] + x0 + self.self_mattn_02(x0,x2,x2)[0]
        x1 = self.self_mattn_10(x1,x0,x0)[0] + x1 + self.self_mattn_12(x1,x2,x2)[0]
        x2 = self.self_mattn_20(x2,x0,x0)[0] + x2 + self.self_mattn_21(x2,x1,x1)[0]
        
        # n, c, d, w, h
        # 将多视图注意力结果转换为n, c, d, w, h
        return x0+x1+x2


class  MVCSBlock(nn.Module):
    def __init__(self, inchannel, outchannel, num_heads, atten):
        super(MVCSBlock, self).__init__()
        '''
        inchannel   
        outchannel  
        num_heads   d
        atten       attention or not
        '''
        self.atten = atten

        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),          
            norm_func(outchannel, affine=True),
            nn.GELU(),
            )

        self.Atten = SADA_Attention(outchannel, num_heads)

        '''
        self.conv_1 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),             
            )
        '''

        self.conv_2 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),          
            norm_func(outchannel, affine=True),
            nn.GELU(),
            )

    def forward(self, x):
        
        x = self.conv_0(x)
        # residual = x
        if self.atten:
            x = self.Atten(x)
        # out = self.conv_1(x)
        return self.conv_2(x) # + residual


class Blocks(nn.Module):
    def __init__(self, inchannel, outchannel, num_heads, atten= [False,False]):
        super(Blocks, self).__init__()
        '''
        
        '''
        self.block0 = MVCSBlock(inchannel, outchannel, num_heads, atten[0])
        self.block1 = MVCSBlock(outchannel, outchannel, num_heads, atten[1])        
        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),          
            norm_func(outchannel, affine=True),
            nn.GELU(),
            )
        self.DropLayer = nn.Dropout(0.2)         

    def forward(self, x):
        # print(x.shape)
        residual = x
        x = self.block0(x)
        x = self.DropLayer(x)
        x = self.block1(x)
        return x + self.conv_0(residual)


class InputUnit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(InputUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)


class MVCSNet(nn.Module):

    def __init__(self,
                 dim,
                 outchannel,
                 inchannel=1,
                 num_classes=1,
                 num_head=[16,8,4,2],
                 drop_rate = 0.2,
                 **kwargs
                 ):       
        super(MVCSNet, self).__init__()
        '''
        inchannel   c
        num_head    d
        '''
        self.num_classes = 1

        base_channel = 64
        num_heads = num_head

        self.A_input = InputUnit(inchannel, base_channel) 
        self.Pooling = nn.AvgPool3d(2, 2) 

        self.A_conv0 = Blocks(base_channel, base_channel*2,num_heads[0], [False, False])

        self.A_conv1 = Blocks(base_channel*2, base_channel*4, num_heads[1], [True, True])
        
        self.A_conv2 = Blocks(base_channel*4, base_channel*8, num_heads[2], [True, True])
          
        # self.A_conv3 = Blocks(base_channel*8, base_channel*8, num_heads[3],[True, True])
        
        self.ClassHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, x):
       
        n, c, d, h, w = x.size()
     
        x0 = self.A_input(x) # n, 64, 32, h, w
        x0 = self.Pooling(x0) # n, 64, 16, h/2, w/2
        x1 = self.A_conv0(x0) # n, 128, 16, h/2, w/2

        x1 = self.Pooling(x1) # n, 128, 8, h/4, w/4
        x2 = self.A_conv1(x1) # n, 256, 8, h/4, w/4

        x3 = self.Pooling(x2) # n, 256, 4, h/8, w/8
        x3 = self.A_conv2(x3) # n, 512, 4, h/8, w/8

        # x4 = self.Pooling(x3) # n, 512, 2, h/16, w/16
        # x4 = self.A_conv3(x4) # n, 512, 2, h/16, w/16

        # n, 512, 1, 1, 1 => n, 512 
        out = self.ClassHead(nn.AdaptiveMaxPool3d((1,1,1))(x4).view(x4.shape[0], -1)) # n, num_classes    

        x4 = x3.reshape(n,512,-1) # n, 512, ?
        return nn.Linear(x4.shape[2],dim)(x4),out # n, 512, dim 


# Cross Attention Implementation
class CrossAttention(nn.Module):

    def __init__(self,dim_i,dim_t,dim,heads=4,dropout=0.2):
        super().__init__()

        self.fi1=nn.Linear(dim_i,dim)
        self.fi2=nn.Linear(dim,dim)
        self.ft1=nn.Linear(dim_t,dim)
        self.ft2=nn.Linear(dim,dim)

        self.conv_i1 = nn.Linear(dim, dim)
        self.conv_i2 = nn.Linear(dim, dim)
        self.conv_i3 = nn.Linear(dim, dim)
        self.conv_t1 = nn.Linear(dim, dim)
        self.conv_t2 = nn.Linear(dim, dim)
        self.conv_t3 = nn.Linear(dim, dim)

        self.self_attn_V = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_attn_T = MultiheadAttention(dim, heads,dropout=dropout)
        
    def forward(self,i,t):
        # i: b, len, dim_i  t: b, len, dim_t
        residual_i = i

        # i: b, len, dim  t: b, len, dim
        i_ = self.fi1(i)
        t_ = self.ft1(t)
        residual_i_ = i_
        residual_t_ = t_

        v1 = self.conv_i1(i_)
        k1 = self.conv_i2(i_)
        q1 = self.conv_i3(i_)
        v2 = self.conv_t1(t_)
        k2 = self.conv_t2(t_)
        q2 = self.conv_t3(t_)

        V_ = self.self_attn_V(q2, k1, v1)[0]
        T_ = self.self_attn_T(q1, k2, v2)[0]
        V_ = V_ + residual_i_
        T_ = T_ + residual_t_

        V_ = self.fi2(V_)
        T_ = self.ft2(T_)

        V_ = V_ + residual_i    # 需要让dim_i = dim

        return torch.cat((V_,T_),1) # b, len, dim
     

# MLP
class TFMLP(nn.Module):
    def __init__(self,dim_t,dim,num_classes=1,heads=4,dropout = 0.2, activation='gelu'):
        super().__init__()

        self.trans = nn.TransformerEncoderLayer(d_model=dim_t, nhead=heads, dim_feedforward=dim, dropout=dropout, activation=activation)
        self.linear1=nn.Linear(dim_t,dim)
        self.relu1=nn.ReLU()
        self.linear2=nn.Linear(dim,dim*2) 
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(dim*2,dim)

        self.dense=nn.Linear(dim,num_classes)

    def forward(self,x):
        
        x = self.trans(x) # expect 3 dims, the size will not change
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x,self.dense(x)


# Ours
class Ours(nn.Module):
    def __init__(self,dim_i,dim_t,dim,num_classes):
        super().__init__()

        self.MVCS = MVCSNet()
        self.TableDetect = TFMLP(dim_t=dim_t,dim=dim)
        self.Fusion = CrossAttention(dim_i=dim_i,dim_t=dim_t,dim=dim)
        self.classifier=nn.Linear(dim,num_classes)
    
    def forward(self,i,t):
        # image (b, c, d, w, h) -> (b, 512, ) (b, num_classes)
        i,pre_i = self.MVCS(i)

        # table (b, len, dim) -> (b, len, dim) (b, len, num_classes)
        t,pre_t = self.TableDetect(table)

        # fusion (b, w, hi) (b, w, ht) -> (b, 2*w, h)
        fusion = self.Fusion(i,t)

        return classifier(fusion)


if __name__ == "__main__":
    cross_atten = CrossAttention(dim_i=512, dim_t=768, dim=512)
    image = torch.randn((4, 38, 512))
    table = torch.randn((4, 38, 768))
    fusion = cross_atten(image, table)
