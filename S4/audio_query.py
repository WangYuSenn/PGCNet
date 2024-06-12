import torch
import torch.nn as nn
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    def forward(self, query, audio_feat):
        out1 = self.self_attn(query, query, query)[0]
        query = self.norm1(query+out1)
        out2 = self.cross_attn(query, audio_feat, audio_feat)[0]
        query = self.norm2(query+out2)
        query = self.norm3(query+query)
        return query
class AttentionGenerator(nn.Module):
    def __init__(self, num_layers, query_num, embed_dim=256, num_heads=4, hidden_dim=512):
        super().__init__()
        self.num_layers = num_layers
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.query = nn.Embedding(query_num, embed_dim)
        self.layers = AttentionLayer(embed_dim, num_heads, hidden_dim)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, audio_feat):
        audio_feat = audio_feat.unsqueeze(1)
        bs = audio_feat.shape[0]
        query = self.query.weight[None, :, :].repeat(bs, 1, 1)
        query = self.layers(query, audio_feat)
        return query