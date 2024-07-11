import torch
import torch.nn.functional as F
from torch import nn, einsum
# import sys;sys.path.append('./')
from cross_atten.sd_cross_atten import CrossAttention
from einops import rearrange, repeat
from cross_atten.corss_ft_transformer import Attention, FeedForward, GEGLU, NumericalEmbedder
from cross_atten.mamba import Mamba, MambaConfig
from cross_atten.jamba import Jamba, JambaLMConfig

class Cross_mamba_both(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0., 
        cross_ff_multi = 2, 
        cross_ff_dropout = 0.1,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer
        config = MambaConfig(d_model=dim, n_layers=depth, use_cuda=True)
        self.transformer = Mamba(config)

        # self.transformer = Transformer(
        #     dim = dim,
        #     depth = depth,
        #     heads = heads,
        #     dim_head = dim_head,
        #     attn_dropout = attn_dropout,
        #     ff_dropout = ff_dropout
        # )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=160*160)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)
        
    def forward(self, x_categ, x_numer, feature_img, image_condition=None):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            mri_condition  = image_condition[0]
            mri_condition = rearrange(mri_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
            pet_condition = image_condition[1]
            pet_condition = rearrange(pet_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
            whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x, feature_img), dim = 1)

        # attend

        x = self.transformer(x)
        x = torch.mean(x, dim = 1, keepdims = True)
        # x = x[:, 0:1]  # x[:, 0] will make less dimension
        x = self.final_cross(x, whole_condition) + x
        x = self.final_feed(x) + x
        
        x = x.squeeze(1) # make less dimension to linear layer

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        return logits
 
class Cross_jamba_both(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0., 
        cross_ff_multi = 2, 
        cross_ff_dropout = 0.1,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer
        config = JambaLMConfig(d_model=dim, n_layers=depth*2, 
                               use_cuda=True, mlp_size=dim*2, 
                               attention_dropout=attn_dropout, num_attention_heads=heads)
        self.transformer = Jamba(config)


        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=160*160)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)
        
    def forward(self, x_categ, x_numer, feature_img, image_condition=None):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            mri_condition  = image_condition[0]
            mri_condition = rearrange(mri_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
            pet_condition = image_condition[1]
            pet_condition = rearrange(pet_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
            whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x, feature_img), dim = 1)

        # attend

        x = self.transformer(x)
        x = torch.mean(x[0], dim = 1, keepdims = True)
        # x = x[:, 0:1]  # x[:, 0] will make less dimension
        x = self.final_cross(x, whole_condition) + x
        x = self.final_feed(x) + x
        
        x = x.squeeze(1) # make less dimension to linear layer

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        return logits
  

class Cross_mamba_ablation(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0., 
        cross_ff_multi = 2, 
        cross_ff_dropout = 0.1,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer
        config = MambaConfig(d_model=dim, n_layers=depth, use_cuda=True)
        self.transformer = Mamba(config)

        # self.transformer = Transformer(
        #     dim = dim,
        #     depth = depth,
        #     heads = heads,
        #     dim_head = dim_head,
        #     attn_dropout = attn_dropout,
        #     ff_dropout = ff_dropout
        # )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=160*160)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)
        
    def forward(self, x_categ, x_numer, feature_img=None, image_condition=None, no_table=False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            mri_condition  = image_condition[0]
            mri_condition = rearrange(mri_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
            pet_condition = image_condition[1]
            pet_condition = rearrange(pet_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
            whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        if no_table:
            x = torch.cat((cls_tokens, feature_img), dim = 1)
        else:
            if feature_img != None:
                x = torch.cat((cls_tokens, x, feature_img), dim = 1)
            else:
                x = torch.cat((cls_tokens, x), dim = 1)
        


        # attend

        x = self.transformer(x)
        x = torch.mean(x, dim = 1, keepdims = True)
        # x = x[:, 0:1]  # x[:, 0] will make less dimension
        if image_condition != None:
            x = self.final_cross(x, whole_condition) + x
            x = self.final_feed(x) + x
        
        x = x.squeeze(1) # make less dimension to linear layer

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        return logits
 