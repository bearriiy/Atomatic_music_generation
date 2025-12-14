# - 包含 PositionalEncoding 类，实现Transformer的位置编码
# - 实现 TransformerMusicModel 类，包含完整的Transformer架构：
# - 编码器-解码器结构
# - 情绪嵌入机制
# - 音乐生成功能

import torch
import torch.nn as nn
import math

# --------------------------
# Transformer模型核心组件
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerMusicModel(nn.Module):
    def __init__(self, vocab_size, num_emotions=3, d_model=128, nhead=8, num_layers=3,
                 dropout=0.3, feedforward_dim=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_emotions = num_emotions

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emotion_embedding = nn.Embedding(num_emotions, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, emotion_id):
        batch_size = src.size(0)
        src = torch.clamp(src, 0, self.vocab_size - 1)
        tgt = torch.clamp(tgt, 0, self.vocab_size - 1)

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        # 添加情绪嵌入
        emotion_emb = self.emotion_embedding(emotion_id).unsqueeze(1)
        src_emb = src_emb + emotion_emb.expand(-1, src_emb.size(1), -1)
        tgt_emb = tgt_emb + emotion_emb.expand(-1, tgt_emb.size(1), -1)

        src_emb = self.dropout(self.pos_encoder(src_emb))
        tgt_emb = self.dropout(self.pos_encoder(tgt_emb))

        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=src.device)

        enc_out = self.encoder(src_emb)
        # 使用tgt的所有位置除了最后一个作为decoder输入
        dec_out = self.decoder(
            tgt_emb[:, :-1],
            enc_out,
            tgt_mask=tgt_mask[:-1, :-1]
        )

        out = self.fc_out(dec_out)
        return out

    def generate(self, src, emotion_id, max_len=200, temperature=1.0):
        self.eval()
        device = src.device
        generated = src.clone()

        with torch.no_grad():
            # 预先计算编码器输出（固定）
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            emotion_emb = self.emotion_embedding(
                torch.tensor([emotion_id], device=device)
            ).unsqueeze(1)
            src_emb = src_emb + emotion_emb.expand(-1, src_emb.size(1), -1)
            src_emb = self.dropout(self.pos_encoder(src_emb))
            enc_out = self.encoder(src_emb)

            for _ in range(max_len - src.size(1)):
                current_seq = generated
                tgt_seq_len = current_seq.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)

                # 目标序列嵌入
                tgt_emb = self.embedding(current_seq) * math.sqrt(self.d_model)
                tgt_emb = tgt_emb + emotion_emb.expand(-1, tgt_emb.size(1), -1)
                tgt_emb = self.dropout(self.pos_encoder(tgt_emb))

                dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
                next_token_logits = dec_out[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        self.train()
        return generated.squeeze(0).cpu().numpy()