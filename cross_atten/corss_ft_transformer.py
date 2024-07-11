import torch
import torch.nn.functional as F
from torch import nn, einsum
# import sys;sys.path.append('./')
from cross_atten.sd_cross_atten import CrossAttention
from einops import rearrange, repeat

# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

class Transformer_cross(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout, 
        dim_cross=None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.cross_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

        for _ in range(depth):
            self.cross_layers.append(nn.ModuleList([
                CrossAttention(n_heads=heads, d_embed=dim, d_cross=dim_cross)
                # FeedForward(dim, dropout= ff_dropout)
            ]))
        
        

    def forward(self, x, return_attn = False, condition=None):
        post_softmax_attns = []

        for (attn, ff), attn_cross in zip(self.layers, self.cross_layers):
            attn_out = attn_cross[0](x, condition)
            x = attn_out + x
            # x = ff_cross(x) + x

            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types, shrink_dim=False):
        super().__init__()
        if shrink_dim: 
            self.linear = nn.Linear(num_numerical_types, num_numerical_types//2)
            num_numerical_types = num_numerical_types // 2
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        if hasattr(self, 'linear'):
            x = self.linear(x)
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

class Cross_transformer_both(nn.Module):
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

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

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

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0:1]  # x[:, 0] will make less dimension
        x = self.final_cross(x, whole_condition) + x
        x = self.final_feed(x) + x
        x = x.squeeze(1) # make less dimension to linear layer

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        return logits
    

class Cross_transformer(nn.Module):
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
        cross_ff_multi = 2
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

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=160*160)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=ff_dropout)
        
    def forward(self, x_categ, x_numer, feature_img, image_condition=None):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            image_condition = rearrange(image_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
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

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0:1]  # x[:, 0] will make less dimension
        x = self.final_cross(x, image_condition) + x
        x = self.final_feed(x) + x
        x = x.squeeze(1) # make less dimension to linear layer

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        return logits
    
class FTTransformer_cross_img(nn.Module):
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
        ff_dropout = 0.
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

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=160*160)

    def forward(self, x_categ, x_numer, feature_img, image_condition=None):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            image_condition = rearrange(image_condition, 'b c h w d -> (b c) (h w) d').transpose(1, 2).contiguous()
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

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0:1]  # x[:, 0] will make less dimension
        x = self.final_cross(x, image_condition).squeeze(1)

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        # if not return_attn:
        #     return logits

        return logits
# main class

class FTTransformer_cross(nn.Module):
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
        ff_dropout = 0. , 
        **kwargs
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

        self.transformer = Transformer_cross(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout, 
            **kwargs
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False, condition=None):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        assert condition != None, f'Cross atten needs a condition'
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
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True, condition=condition)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns

if __name__ == "__main__":
    image = torch.randn((1, 16, 20*20*12))
    image_model = nn.Linear(20*20*12, 20*20*12)
    image = image_model(image)
    table = torch.randn((1, 38, 768))
    model = Transformer_cross(768, 6, 8, 16, 0., 0., 20*20*12)
    output = model(table, condition=image)
    loss = output.sum()
    loss.backward()
    for name, param in image_model.named_parameters():
        # 如果参数没有梯度，打印相关信息
        if param.grad is None:
            print(f"Parameter {name} has no gradient.")