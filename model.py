import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class _ConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.LayerNorm([out_planes, h, w]),
            nn.ReLU(inplace=True)
        )

class TVConv(nn.Module):
    def __init__(self, channels, h, w, TVConv_k=3, stride=1, TVConv_posi_chans=4, 
                 TVConv_inter_chans=64, TVConv_inter_layers=3, TVConv_Bias=False):
        super(TVConv, self).__init__()
        self.TVConv_k = TVConv_k
        self.TVConv_k_square = TVConv_k**2
        self.channels = channels
        self.h = h
        self.w = w

        out_chans = self.TVConv_k_square * self.channels

        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)

        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        
        self.bias_layers = None
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)

        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(inter_chans, out_chans, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape

        weight = self.weight_layers(self.posi_map) # (1, C*k^2, H, W)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w)

        out = self.unfold(x).view(b, self.channels, self.TVConv_k_square, h, w)

        out = (weight * out).sum(dim=2)
        
        if self.bias_layers is not None:
            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out

class ACFM(nn.Module):
    def __init__(self, dim, h, w, num_heads=4, bias=False):
        super(ACFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.local_branch = TVConv(channels=dim, h=h, w=w, TVConv_k=3)

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        x_3d = x.unsqueeze(2) # (B, C, 1, H, W)
        qkv = self.qkv_dwconv(self.qkv(x_3d))
        qkv_2d = qkv.squeeze(2) # (B, 3*C, H, W)
        
        q, k, v = qkv_2d.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_global = (attn @ v)
        out_global = rearrange(out_global, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        out_global = self.project_out(out_global)

        out_local = self.local_branch(x)

        return x + out_global + out_local

class SSFE(nn.Module):
    def __init__(self, input_channels, out_channels, h, w):

        super(SSFE, self).__init__()

        self.spatial_project = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.spatial_acfm = ACFM(dim=out_channels, h=h, w=w, num_heads=4)

        self.spectral_branch = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 1, 1), padding=(3, 0, 0)), 
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, out_channels, kernel_size=(input_channels, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.spa_weight = nn.Parameter(torch.ones(1)) 
        self.spe_weight = nn.Parameter(torch.ones(1))

        hidden = max(16, out_channels // 4)
        self.spec_gate_net = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels),
            nn.Sigmoid()
        )
        
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0) 
        
        freq_len = (input_channels // 2) + 1  # rfft length for spectral axis
        hidden_f = max(64, freq_len)
        self.specfreq_mlp = nn.Sequential(
            nn.Linear(freq_len, hidden_f),
            nn.ReLU(),
            nn.LayerNorm(hidden_f),
            nn.Linear(hidden_f, out_channels)
        )
        self.spatialfreq_mlp = nn.Sequential(
            nn.Linear(input_channels, max(64, input_channels)),
            nn.ReLU(),
            nn.LayerNorm(max(64, input_channels)),
            nn.Linear(max(64, input_channels), out_channels)
        )
        # start fusion weight at 0 (gradual/controlled fusion)
        self.specfreq_weight = nn.Parameter(torch.tensor(0.0))

        self.relu = nn.ReLU()
        

    def forward(self, x):
        # x: (B, 1, C, H, W)
        b = x.shape[0]

        x_spatial_in = x.squeeze(1)   # (B, C, H, W)
        x_spa_4D = self.spatial_project(x_spatial_in) 
        x_spa_4D = self.spatial_acfm(x_spa_4D)        # -> (B, D, H, W)

        x_spectral = self.spectral_branch(x) 
        x_spe_4D = x_spectral.squeeze(2)               # -> (B, D, H, W)

        x_orig = x.squeeze(1)
        b, C, H, W = x_orig.shape

        scale_factors = [1.0, 0.5, 0.25]

        total_importance = x_orig.new_zeros(b, C)  # (B, C)

        weights = F.softmax(self.scale_weights, dim=0)

        for i, scale in enumerate(scale_factors):

            if scale == 1.0:
                x_scaled = x_orig  # (B, C, H, W)
            else:
                h_s = max(1, int(H * scale))
                w_s = max(1, int(W * scale))

                x_scaled = F.adaptive_avg_pool2d(
                    x_orig,
                    (h_s, w_s)
                )

            fft_freq = torch.fft.rfft2(x_scaled,dim=(-2, -1))  # (B, C, h_s, w_freq)

            psd = fft_freq.abs().pow(2)

            fft_vec = psd.view(b, C, -1)  # (B, C, N_freq)

            fft_norm = F.normalize(fft_vec,p=2,dim=-1,eps=1e-8)

            freq_corr = torch.bmm(fft_norm,fft_norm.transpose(1, 2))  # (B, C, C)

            scale_importance = torch.sum(torch.abs(freq_corr),dim=2)  # (B, C)

            total_importance = (total_importance + weights[i] * scale_importance)

        total_importance = self.spatialfreq_mlp(total_importance)  # (B, D)

        x_orig = x.squeeze(1)
        spec_fft = torch.fft.rfft(x_orig, dim=1)               # (B, freq_len, H, W)
        spec_psd = (spec_fft.abs() ** 2)                       # (B, freq_len, H, W)
        spec_energy = spec_psd.mean(dim=(-2, -1))              # (B, freq_len)

        spec_energy = torch.log1p(spec_energy)                 # reduce dynamic range
        spec_energy = F.layer_norm(spec_energy, spec_energy.shape[-1:])  # (B, freq_len)

        specfreq_feats = self.specfreq_mlp(spec_energy)       # (B, D)
        specfreq_feats = F.normalize(specfreq_feats, p=2, dim=-1, eps=1e-8)

        total_norm = F.layer_norm(total_importance, total_importance.shape[-1:])
        alpha = torch.sigmoid(self.specfreq_weight)  # (0,1)
        fused = (1.0 - alpha) * total_norm + alpha * specfreq_feats
        total_importance = fused  # (B, D)

        spec_gate_raw = self.spec_gate_net(total_importance)  # (B, D) in (0,1)
        spec_gate = (2.0 * spec_gate_raw - 1.0) * 0.5  # mapped to (-0.5, 0.5)

        x_fused_4D = (self.spa_weight * x_spa_4D) + (self.spe_weight * x_spe_4D)
        
        return self.relu(x_fused_4D), spec_gate


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class TSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., gate_alpha=0.2):
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        self.gate_alpha = gate_alpha

    def forward(self, x, spec_gate=None):

        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        b, h, N, d = w.shape  # d = D // heads

        if spec_gate is not None:
            # spec_gate: (B, D) -> (B, H, 1, d)
            gate = rearrange(spec_gate, 'b (h d) -> b h 1 d', h=self.heads)
            # scale factor in [1-alpha, 1+alpha]
            w = w * (1.0 + self.gate_alpha * gate)

        w_normed = F.normalize(w, dim=-2)
        w_sq = w_normed ** 2
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) 
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)
        out = - torch.mul(w * Pi.unsqueeze(-1), attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerWithTSSA(nn.Module):
    def __init__(self, dim, depth=2, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TSSA(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))
    def forward(self, x, spec_gate=None):
        """
        x: (B, N, D)
        spec_gate: None or (B, D) -> passed to each TSSA layer
        """
        for attn, ff in self.layers:
            x = x + attn(x, spec_gate=spec_gate)
            x = x + ff(x)
        return x

class falcon(nn.Module):
    def __init__(self, num_classes=9, token_dim=64, transformer_depth=2, hs_channels=30):
        super().__init__()
        
        self.feature_extractor = SSFE(input_channels=hs_channels, out_channels=token_dim, h=13, w=13)
        
        self.pos_emb = nn.Parameter(torch.randn(1, token_dim, 1, 1))

        self.transformer = TransformerWithTSSA(dim=token_dim, depth=transformer_depth)
        

        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, num_classes)
        )

    def forward(self, x):
        x_fused_4D, spec_gate = self.feature_extractor(x) # -> (B, D, H, W), (B, D)
        
        if self.pos_emb.shape[-2:] != x_fused_4D.shape[-2:]:
            pos_emb = F.interpolate(self.pos_emb, size=x_fused_4D.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pos_emb = self.pos_emb
            
        x_transformer_in_4D = x_fused_4D + pos_emb
        x_transformer_in = rearrange(x_transformer_in_4D, 'b c h w -> b (h w) c') # -> (B, N, D)
        
        x_fused_token = self.transformer(x_transformer_in, spec_gate=spec_gate) # -> (B, N, D)

        x_refined_token = x_fused_token + x_transformer_in
        
        x_final = x_refined_token.mean(dim=1)
        logits = self.head(x_final)
        
        return logits
