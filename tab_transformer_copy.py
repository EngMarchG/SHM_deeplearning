import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Use torch.mean instead of .mean() for better performance
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps) * self.scale

class Residual(nn.Module):
    def __init__(self, fn, dim, num_landmarks=15):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)
        self.act = nn.SiLU()
        self.num_landmarks = num_landmarks
        self.groups = 4

        # Grouped convolutions
        self.conv1 = nn.Conv2d(self.groups, self.groups, kernel_size=(dim // num_landmarks, 1), dilation=3, groups=self.groups)
        self.conv2 = nn.Conv2d(self.groups, self.groups, kernel_size=(dim // num_landmarks, 1), dilation=3, groups=self.groups)
        self.conv3 = nn.Conv2d(self.groups, self.groups, kernel_size=(dim // num_landmarks, 1), dilation=3, groups=self.groups)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, num_landmarks))

        # Short skip connection
        self.short_skip = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        
        # Long skip connection using 2D convolutions
        self.long_skip = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(dim, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(dim, 1))
        )

        # Mid skip connection
        self.mid_skip = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

        # Gated skip connection
        self.gated_skip = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.Sigmoid()
        )

        # Downstream processing
        self.downstream = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, **kwargs):
        short_residual = self.short_skip(x.transpose(1, 2)).transpose(1, 2)
        
        # Prepare for long skip connection
        long_residual = self.long_skip(x.unsqueeze(1)).squeeze(1)
        
        out = self.fn(x, **kwargs)
        
        # Early skip point
        if out.shape == x.shape:
            out = out + 0.2 * short_residual + 0.2 * long_residual
            return self.act(self.norm(out))
        
        # Mid-skip connection
        mid_residual = self.mid_skip(out.transpose(1, 2)).transpose(1, 2)
        
        out = out.view(-1, self.groups, out.shape[1] // self.num_landmarks, self.num_landmarks)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool(out)
        
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        
        # Gated skip connection
        gated_residual = self.gated_skip(x.transpose(1, 2)).transpose(1, 2)
        
        # Combine with downstream processed data and all skip connections
        downstream = self.downstream(out)
        out = 0.4 * out + 0.3 * downstream + 0.1 * short_residual + 0.1 * long_residual + 0.05 * mid_residual + 0.05 * gated_residual
        
        out = self.act(self.norm(out))
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        freqs = torch.einsum('i,j->ij', t.float(), self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    q = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q, k


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 16, dropout = 0., num_key_value_heads = 4):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.num_key_value_heads = num_key_value_heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * num_key_value_heads * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(dim_head)

    def forward(self, x):
        b, n, _, h, kv_h = *x.shape, self.heads, self.num_key_value_heads
        q = self.to_q(x).reshape(b, n, h, -1)
        kv = self.to_kv(x).reshape(b, n, kv_h * 2, -1)
        k, v = kv.chunk(2, dim = 2)

        q = q.reshape(b, n, h, -1)
        k = k.reshape(b, n, kv_h, -1)
        v = v.reshape(b, n, kv_h, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        freqs = self.rotary_emb(torch.arange(n, device=x.device))
        q, k = apply_rotary_pos_emb(q, k, freqs)

        q = repeat(q, 'b (h g) n d -> b h n d', g = h // kv_h)
        k = repeat(k, 'b h n d -> b (h g) n d', g = h // kv_h)
        v = repeat(v, 'b h n d -> b (h g) n d', g = h // kv_h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SparseAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 16, dropout = 0., num_landmarks = 16):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.attention = GroupedQueryAttention(dim, heads, dim_head, dropout)

    def forward(self, x):
        b, n, d = x.shape
        landmarks = x[:, ::n//self.num_landmarks]
        
        attn_landmarks = self.attention(x)
        attn_between_landmarks = self.attention(landmarks)
        
        out = attn_landmarks + torch.einsum('bnl,blm->bnm', attn_landmarks, attn_between_landmarks)
        return out


class MoE(nn.Module):
    def __init__(self, dim, num_experts = 4, hidden_dim = None, activation = nn.SiLU, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        hidden_dim = default(hidden_dim, dim * 4)

        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        b, n, d = x.shape
        g = self.gate(x)
        weights = F.softmax(g, dim=-1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        out = torch.einsum('bne,bnec->bnc', weights, expert_outputs)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SENet(nn.Module):
    def __init__(self, channel, reduction=5):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, num_landmarks):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.gap = nn.AdaptiveAvgPool1d(1)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, SparseAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, num_landmarks=num_landmarks)), dim),
                SENet(dim),
                Residual(PreNorm(dim, MoE(dim, dropout=ff_dropout)), dim),
                SENet(dim),
                Residual(PreNorm(dim, nn.Sequential(
                    SparseAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, num_landmarks=num_landmarks),
                    nn.Conv1d(dim, dim, kernel_size=3, dilation=3)
                )), dim),
                Residual(PreNorm(dim, nn.Sequential(
                    SparseAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, num_landmarks=num_landmarks),
                    nn.Conv1d(dim, dim, kernel_size=3, dilation=3)
                )), dim),
                SENet(dim),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout)), dim),
                SENet(dim),
            ]))

    def forward(self, x, return_attn=False):
        for layer in self.layers:
            for sublayer in layer:
                if isinstance(sublayer, Residual):
                    x = sublayer(x)
                elif isinstance(sublayer, SENet):
                    x = sublayer(x)
                else:
                    x = sublayer(x.transpose(1, 2)).transpose(1, 2)

        x = self.gap(x.transpose(1, 2)).squeeze(-1)

        return x


class MLP(nn.Module):
    def __init__(self, dims, act = None, dropout=0., use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.BatchNorm1d(dim_out))  # Batch normalization

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.mlp:
            if self.use_residual and isinstance(layer, nn.Linear):
                x = F.relu(x + layer(x))  # Add skip connection
            else:
                x = layer(x)
        return x


# main class
class TabTransformer_edit(nn.Module):
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
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        mlp_dropout = 0.,
        mlp_use_residual = False,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.,   # in paper, they reserve dimension / 8 for category shared embedding
        use_layer_norm=True,
        layer_norm_eps=1e-5,
        autoencoder_dim = 15
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.continuous_layer_norm = nn.LayerNorm(num_continuous, eps=layer_norm_eps)

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        # take care of shared category embed
        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.norm = nn.LayerNorm(num_continuous)


        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            num_landmarks=dim_out
        )

        # mlp to logits
        input_size = (dim * self.num_categories) + num_continuous
        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act, dropout=mlp_dropout, use_residual=mlp_use_residual)


    def forward(self, x_categ, x_cont, return_attn = False):
        xs = []

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_continuous > 0:
            if self.use_layer_norm:
                x_cont = self.continuous_layer_norm(x_cont)
                
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            x, attns = self.transformer(categ_embed, return_attn = True)

            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim = -1)

        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns
