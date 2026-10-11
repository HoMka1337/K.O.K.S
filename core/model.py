import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe.to(x.device)
        return x + pe[:x.size(1)]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, d_model = x.size()
        device = x.device
        
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class CodexModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.prompt_style = "agent"
        self.generation_ready = True
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits
    
    def generate(self, tokenizer, prompt: str, max_new_tokens: int = 80, temperature: float = 0.85, top_k: int = 50) -> str:
        self.eval()
        device = next(self.parameters()).device
        input_ids = tokenizer.encode_prompt(prompt)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_ids.copy()
        eos_token = tokenizer.vocab.get('<EOS>', 3)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = generated[-self.max_len:]
                input_tensor = torch.tensor([context], dtype=torch.long).to(device)
                logits = self.forward(input_tensor)
                next_logits = logits[0, -1, :] / temperature
                
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_logits, min(top_k, len(next_logits)))
                    mask = torch.ones_like(next_logits) * float('-inf')
                    mask[top_k_indices] = top_k_values
                    next_logits = mask
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == eos_token:
                    break
                generated.append(next_token)
        
        new_tokens = generated[len(input_ids):]
        result = tokenizer.decode(new_tokens)
        result = self._clean_response(result, prompt)
        return result
    
    def _clean_response(self, text: str, prompt: str) -> str:
        # Structured actions are validated and guarded by the planner/executor.
        try:
            action = json.loads(text)
            if isinstance(action, dict) and ("tool" in action or "done" in action):
                return text.strip()
        except (ValueError, TypeError):
            pass
        lines = text.split('\n')
        clean_lines = []
        skip_phrases = ['пользователь:', 'ассистент:', 'ты - ai-ассистент', 'отвечай кратко', 'доступные инструменты', 'read_file', 'write_file', 'list_files', 'execute_command']
        for line in lines:
            line = line.strip()
            if not line:
                continue
            skip = False
            for phrase in skip_phrases:
                if phrase in line.lower():
                    skip = True
                    break
            if not skip:
                clean_lines.append(line)
        if not clean_lines:
            return "Я не понял запрос."
        result = ' '.join(clean_lines)
        if len(result) > 300:
            result = result[:300] + '...'
        return result