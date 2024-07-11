import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

class SingleConv_modi(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8,
                 padding=1, dropout_prob=0.1, is3d=True, stride=1):
        super(SingleConv_modi, self).__init__()

        for name, module in create_conv_stride(in_channels, out_channels, kernel_size, order,
                                        num_groups, padding, dropout_prob, is3d, stride):
            self.add_module(name, module)


  
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=stride,stride=stride,padding=0)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x
        
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = [patch_size, patch_size, patch_size]
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1=patch_size
        # stride2=[patch_size[0]//2,patch_size[1]//2,patch_size[2]//2]
        self.proj1 = project(in_chans,embed_dim,stride1,1,nn.GELU,nn.LayerNorm,True)
        # self.proj2 = project(embed_dim//2,embed_dim,stride2,1,nn.GELU,nn.LayerNorm,True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)  # B C Ws Wh Ww
        # x = self.proj2(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x


def create_conv_stride(in_channels, out_channels, kernel_size, order, num_groups, padding,
                dropout_prob, is3d, stride):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'
    
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        elif char == 'd':
            modules.append(('dropout', nn.Dropout(p=dropout_prob)))
        elif char == 'D':
            modules.append(('dropout2d', nn.Dropout2d(p=dropout_prob)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']")

    return modules


class Combine_classfier(nn.Module):
    def __init__(self, latent_dim, f_maps=(64, 128, 256)):
        super(Combine_classfier, self).__init__()
        module_list = []        
        module_list.append(SingleConv_modi(f_maps[0]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=2))
        module_list.append(SingleConv_modi(f_maps[1]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=1))
        self.extract_module = nn.ModuleList(module_list)


        self.feature3_conv = SingleConv_modi(latent_dim*2, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature4_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature5_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        
        self.max_pooling = nn.MaxPool3d(3, 2)
        self.avg_pooling = nn.AvgPool3d(3, 2)
        # self.fully_con = nn.Sequential(*[nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 1)])
        self.fully_con = nn.Sequential(*[nn.Linear(512, 1)])
        
        
    def forward(self, encoders_feature, decoders_feature):
        assert len(encoders_feature) == len(decoders_feature) == 2
        concate_features = []
        for en_feature, de_feature, module in zip(encoders_feature, decoders_feature, self.extract_module):
            concat_f = torch.cat([en_feature, de_feature], dim=1) # concate on the channel dimension
            concate_features.append(module(concat_f))
        whole = self.max_pooling(torch.cat(concate_features, dim=1))
        f3 = self.feature3_conv(whole)
        f3 = self.max_pooling(f3)
        f4 = self.feature4_conv(f3)
        f4 = self.max_pooling(f4)
        f5 = self.feature5_conv(f4)
        f5 = self.avg_pooling(f5)
        # descs = F.normalize(torch.cat((f5, (f5 ** 2) - 1), dim=-1), dim=-1, p=2)
        descs = f5
        feats = descs.view(-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4])
        # feats = F.normalize(descs.view(-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4]), dim=-1, p=2)
        output = self.fully_con(feats)
        return output
        

class Combine_classfier_cross(nn.Module):
    def __init__(self, latent_dim, f_maps=(64, 128, 256)):
        super(Combine_classfier_cross, self).__init__()
        module_list = []        
        module_list.append(SingleConv_modi(f_maps[0]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=2))
        module_list.append(SingleConv_modi(f_maps[1]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=1))
        self.extract_module = nn.ModuleList(module_list)


        self.feature3_conv = SingleConv_modi(latent_dim*2, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature4_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        # self.feature5_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        
        self.max_pooling = nn.MaxPool3d(3, 2, 1)
        self.avg_pooling = nn.AvgPool3d(3, 2, 1)
        # self.fully_con = nn.Sequential(*[nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 1)])
        # self.fully_con = nn.Sequential(*[nn.Linear(512, 1)])
        
        
    def forward(self, encoders_feature, decoders_feature):
        assert len(encoders_feature) == len(decoders_feature) == 2
        concate_features = []
        for en_feature, de_feature, module in zip(encoders_feature, decoders_feature, self.extract_module):
            concat_f = torch.cat([en_feature, de_feature], dim=1) # concate on the channel dimension
            concate_features.append(module(concat_f))
        whole = self.max_pooling(torch.cat(concate_features, dim=1)) # whole.shape: (b, 32, 40, 40, 24)
        f3 = self.feature3_conv(whole)
        f3 = self.max_pooling(f3) # f3.shape: (1, 16, 20, 20, 12)
        f4 = self.feature4_conv(f3)
        f4 = self.max_pooling(f4)
        output = f4.view(-1, f4.shape[1], f4.shape[2]*f4.shape[3]*f4.shape[4])
        # f5 = self.feature5_conv(f4)
        # f5 = self.avg_pooling(f5)
        # descs = f5
        # feats = descs.view(-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4])
        # output = self.fully_con(feats)
        return output


class Combine_classfier_emb(nn.Module):
    def __init__(self, latent_dim, f_maps=(64, 128, 256)):
        super(Combine_classfier_emb, self).__init__()
        module_list = []        
        module_list.append(SingleConv_modi(f_maps[0]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=2))
        module_list.append(SingleConv_modi(f_maps[1]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=1))
        self.extract_module = nn.ModuleList(module_list)
        self.patch_embbeder = PatchEmbed(patch_size=16, in_chans=latent_dim*2, embed_dim=512, norm_layer=nn.LayerNorm)

        
        
    def forward(self, encoders_feature, decoders_feature):
        assert len(encoders_feature) == len(decoders_feature) == 2
        concate_features = []
        for en_feature, de_feature, module in zip(encoders_feature, decoders_feature, self.extract_module):
            concat_f = torch.cat([en_feature, de_feature], dim=1) # concate on the channel dimension
            concate_features.append(module(concat_f))
        whole = torch.cat(concate_features, dim=1)
        output = self.patch_embbeder(whole)
        
        return output.flatten(2).transpose(1, 2).contiguous()

class Combine_classfier_(nn.Module):
    def __init__(self, latent_dim, f_maps=(64, 128, 256)):
        super(Combine_classfier_, self).__init__()
        module_list = []        
        module_list.append(SingleConv_modi(f_maps[0]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=2))
        module_list.append(SingleConv_modi(f_maps[1]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=1))
        self.extract_module = nn.ModuleList(module_list)


        self.feature3_conv = SingleConv_modi(latent_dim*2, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature4_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature5_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        
        self.max_pooling = nn.MaxPool3d(3, 2)
        self.avg_pooling = nn.AvgPool3d(3, 2)
        # self.fully_con = nn.Sequential(*[nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 1)])
        self.fully_con = nn.Sequential(*[nn.Linear(1024, 1)])
        
        
    def forward(self, encoders_feature, decoders_feature, ft_feature):
        assert len(encoders_feature) == len(decoders_feature) == 2
        concate_features = []
        for en_feature, de_feature, module in zip(encoders_feature, decoders_feature, self.extract_module):
            concat_f = torch.cat([en_feature, de_feature], dim=1) # concate on the channel dimension
            concate_features.append(module(concat_f))
        whole = self.max_pooling(torch.cat(concate_features, dim=1))
        f3 = self.feature3_conv(whole)
        f3 = self.max_pooling(f3)
        f4 = self.feature4_conv(f3)
        f4 = self.max_pooling(f4)
        f5 = self.feature5_conv(f4)
        f5 = self.avg_pooling(f5)
        # descs = F.normalize(torch.cat((f5, (f5 ** 2) - 1), dim=-1), dim=-1, p=2)
        descs = f5
        b = descs.shape[0]
        feats = descs.view(-1, 1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4])
        Combine_feats = torch.cat((feats, ft_feature), dim=1).view(b,-1)
        # feats = F.normalize(descs.view(-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4]), dim=-1, p=2)
        output = self.fully_con(Combine_feats)
        return output
    
class Combine_classfier_2(nn.Module):
    def __init__(self, latent_dim, f_maps=(64, 128, 256)):
        super(Combine_classfier_2, self).__init__()
        module_list = []        
        module_list.append(SingleConv_modi(f_maps[0]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=2))
        module_list.append(SingleConv_modi(f_maps[1]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=1))
        self.extract_module = nn.ModuleList(module_list)


        self.feature3_conv = SingleConv_modi(latent_dim*2, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature4_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature5_conv = SingleConv_modi(latent_dim, latent_dim, 1, 'cge', 8, 0, is3d=True, stride=1)
        
        self.max_pooling = nn.MaxPool3d(3, 2)
        self.avg_pooling = nn.AvgPool3d(3, 2)
        # self.fully_con = nn.Sequential(*[nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 1)])
        # self.fully_con = nn.Sequential(*[nn.Linear(512, 1)])
        
        
    def forward(self, encoders_feature, decoders_feature):
        assert len(encoders_feature) == len(decoders_feature) == 2
        concate_features = []
        for en_feature, de_feature, module in zip(encoders_feature, decoders_feature, self.extract_module):
            concat_f = torch.cat([en_feature, de_feature], dim=1) # concate on the channel dimension
            concate_features.append(module(concat_f))
        whole = self.max_pooling(torch.cat(concate_features, dim=1)) 
        f3 = self.feature3_conv(whole)
        f3 = self.max_pooling(f3) 
        f4 = self.feature4_conv(f3)
        f4 = self.max_pooling(f4)
        f5 = self.feature5_conv(f4)
        f5 = self.avg_pooling(f5)
        # descs = F.normalize(torch.cat((f5, (f5 ** 2) - 1), dim=-1), dim=-1, p=2)
        descs = f5
        feats = descs.view(-1, 1,  descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4])
        # feats = F.normalize(descs.view(-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4]), dim=-1, p=2)
        return feats
    
class Combine_classfier_vit_mid(nn.Module):
    def __init__(self, seq_length=1):
        super(Combine_classfier_vit_mid, self).__init__()
        self.vit_mid_linear = nn.Linear(320*120, seq_length)
        
    def forward(self, mid_input, mid_output):
            mid_feature = self.vit_mid_linear(rearrange(torch.cat([mid_input, mid_output], dim=1), 'b c h w -> b c (h w)'))
            # mid_feature = rearrange(mid_feature, 'b c 1 -> b 1 c')
            mid_feature = mid_feature.transpose(1, 2).contiguous()
            return mid_feature

class Combine_classfier_3(nn.Module):
    def __init__(self, latent_dim=128, f_maps=(64, 128, 256)):
        super(Combine_classfier_3, self).__init__()
        module_list = []        
        module_list.append(SingleConv_modi(f_maps[0]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=2))
        module_list.append(SingleConv_modi(f_maps[1]*2, latent_dim, 3, 'cge', 8, 1, is3d=True, stride=1))
        self.extract_module = nn.ModuleList(module_list)


        self.feature3_conv = SingleConv_modi(latent_dim*2, latent_dim*2, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature4_conv = SingleConv_modi(latent_dim*2, latent_dim*4, 1, 'cge', 8, 0, is3d=True, stride=1)
        self.feature5_conv = SingleConv_modi(latent_dim*4, latent_dim*4, 1, 'cge', 8, 0, is3d=True, stride=1)
        
        self.max_pooling = nn.MaxPool3d(3, 2)
        self.avg_pooling = nn.AvgPool3d(3, 2)
        # self.fully_con = nn.Sequential(*[nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 1)])
        # self.fully_con = nn.Sequential(*[nn.Linear(512, 1)])
        
        
    def forward(self, encoders_feature, decoders_feature):
        assert len(encoders_feature) == len(decoders_feature) == 2
        concate_features = []
        for en_feature, de_feature, module in zip(encoders_feature, decoders_feature, self.extract_module):
            concat_f = torch.cat([en_feature, de_feature], dim=1) # concate on the channel dimension
            concate_features.append(module(concat_f))
        whole = self.max_pooling(torch.cat(concate_features, dim=1)) 
        f3 = self.feature3_conv(whole)
        f3 = self.max_pooling(f3) 
        f4 = self.feature4_conv(f3)
        f4 = self.max_pooling(f4)
        f5 = self.feature5_conv(f4)
        f5 = self.avg_pooling(f5)
        # descs = F.normalize(torch.cat((f5, (f5 ** 2) - 1), dim=-1), dim=-1, p=2)
        descs = f5
        feats = descs.flatten(2).transpose(1, 2).contiguous()
        return feats
        

        
if __name__ == "__main__":
    a = torch.randn((2, 1, 160, 160, 96))
    import sys; sys.path.append('./')
    from pytorch3dunet.unet3d.model import Residual_mid_UNet3D
    
    model = Residual_mid_UNet3D(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    # model.load_state_dict(torch.load('weights/exp_2_10/model_save/model.pt'))
    cla = Combine_classfier_emb(32)
    list1, list2, _ = model(a, output_mid=True)
    
    b = cla(list1, list2)
    print(b.shape)