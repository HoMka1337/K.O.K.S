
import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000
    ):
        super().__init__()

        if d_model <= 0 or max_len <= 0:
            raise ValueError("Invalid positional encoding dimensions")

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        angles = position * div_term

        pe[:, 0::2] = torch.sin(angles)

        if d_model > 1:
            pe[:, 1::2] = torch.cos(
                angles[:, :pe[:, 1::2].shape[1]]
            )

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)

        if seq_len > self.pe.size(0):
            raise ValueError(
                "Sequence exceeds positional encoding length"
            )

        positions = self.pe[:seq_len].to(
            device=x.device,
            dtype=x.dtype
        )

        return x + positions


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.wk = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.wv = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.out = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len, d_model = x.shape

        q = self.wq(x).reshape(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = self.wk(x).reshape(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = self.wv(x).reshape(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        causal_mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device
        ).triu(diagonal=1)

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(scores.dtype).min
        )

        if mask is not None:
            mask = mask.to(device=x.device)

            if mask.dtype == torch.bool:
                scores = scores.masked_fill(
                    mask,
                    torch.finfo(scores.dtype).min
                )
            else:
                scores = scores + mask

        attention = F.softmax(
            scores.float(),
            dim=-1
        ).to(dtype=v.dtype)

        attention = self.dropout(attention)

        output = torch.matmul(
            attention,
            v
        )

        output = output.transpose(
            1,
            2
        ).contiguous().reshape(
            batch_size,
            seq_len,
            d_model
        )

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            d_model,
            d_ff
        )

        self.fc2 = nn.Linear(
            d_ff,
            d_model
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = CausalSelfAttention(
            d_model,
            n_heads,
            dropout
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(
            x,
            mask
        )

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        feedforward_output = self.ff(x)

        x = self.norm2(
            x + self.dropout(feedforward_output)
        )

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

        vocab_size = int(vocab_size)
        d_model = int(d_model)
        n_heads = int(n_heads)
        n_layers = int(n_layers)
        d_ff = int(d_ff)
        max_len = int(max_len)

        if min(
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_len
        ) <= 0:
            raise ValueError(
                "Model dimensions must be positive"
            )

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "Dropout must be between 0 and 1"
            )

        self.prompt_style = "agent"
        self.generation_ready = True

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_encoding = PositionalEncoding(
            d_model,
            max_len
        )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                d_ff,
                dropout
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(
            d_model,
            vocab_size
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None
    ):
        if x.ndim != 2:
            raise ValueError(
                "Input tensor must have shape [batch, sequence]"
            )

        if x.size(1) == 0:
            raise ValueError(
                "Input sequence cannot be empty"
            )

        if x.size(1) > self.max_len:
            raise ValueError(
                "Input exceeds maximum sequence length"
            )

        x = x.long()

        hidden = self.token_embedding(x)

        hidden = hidden * math.sqrt(
            self.d_model
        )

        hidden = self.pos_encoding(hidden)
        hidden = self.dropout(hidden)

        for layer in self.layers:
            hidden = layer(
                hidden,
                mask
            )

        hidden = self.ln_f(hidden)

        return self.fc_out(hidden)

    def generate(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.85,
        top_k: int = 50
    ) -> str:
        if not self.generation_ready:
            return ""

        if not isinstance(prompt, str):
            raise TypeError(
                "Prompt must be a string"
            )

        if not prompt.strip():
            return ""

        max_new_tokens = max(
            1,
            int(max_new_tokens)
        )

        temperature = float(temperature)

        if not math.isfinite(temperature):
            temperature = 0.85

        temperature = max(
            0.01,
            temperature
        )

        top_k = max(
            0,
            int(top_k)
        )

        self.eval()

        device = next(
            self.parameters()
        ).device

        if hasattr(tokenizer, "encode_prompt"):
            input_ids = tokenizer.encode_prompt(prompt)
        else:
            input_ids = tokenizer.encode(
                prompt,
                add_eos=False
            )

        if not input_ids:
            return ""

        input_ids = [
            int(token_id)
            for token_id in input_ids
        ]

        if any(
            token_id < 0
            or token_id >= self.vocab_size
            for token_id in input_ids
        ):
            raise ValueError(
                "Prompt contains token IDs outside the model vocabulary"
            )

        if len(input_ids) > self.max_len:
            input_ids = (
                [input_ids[0]]
                + input_ids[-(self.max_len - 1):]
                if self.max_len > 1
                else input_ids[-1:]
            )

        generated = list(input_ids)

        eos_token = tokenizer.vocab.get(
            "<EOS>",
            3
        )

        pad_token = tokenizer.vocab.get(
            "<PAD>",
            0
        )

        bos_token = tokenizer.vocab.get(
            "<BOS>",
            2
        )

        invalid_tokens = {
            pad_token,
            bos_token
        }

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = generated[
                    -self.max_len:
                ]

                input_tensor = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=device
                )

                logits = self.forward(
                    input_tensor
                )

                next_logits = (
                    logits[0, -1, :]
                    .float()
                    .clone()
                )

                for token_id in invalid_tokens:
                    if 0 <= token_id < next_logits.numel():
                        next_logits[token_id] = float("-inf")

                next_logits = (
                    next_logits / temperature
                )

                if top_k > 0:
                    effective_k = min(
                        top_k,
                        next_logits.numel()
                    )

                    values, indices = torch.topk(
                        next_logits,
                        effective_k
                    )

                    filtered = torch.full_like(
                        next_logits,
                        float("-inf")
                    )

                    filtered.scatter_(
                        0,
                        indices,
                        values
                    )

                    next_logits = filtered

                probabilities = F.softmax(
                    next_logits,
                    dim=-1
                )

                if (
                    not torch.isfinite(
                        probabilities
                    ).all().item()
                    or probabilities.sum().item() <= 0
                ):
                    next_token = eos_token
                else:
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1
                    ).item()

                if next_token == eos_token:
                    break

                generated.append(next_token)

        new_tokens = generated[
            len(input_ids):
        ]

        if not new_tokens:
            return ""

        response = tokenizer.decode(
            new_tokens
        )

        return self._clean_response(
            response,
            prompt
        )

    def _clean_response(
        self,
        text: str,
        prompt: str
    ) -> str:
        if not isinstance(text, str):
            return ""

        text = text.replace("\x00", "").strip()

        if not text:
            return ""

        try:
            action = json.loads(text)

            if isinstance(action, dict):
                if "tool" in action or "done" in action:
                    return text

        except (ValueError, TypeError):
            pass

        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()

        lines = text.splitlines()
        cleaned = []

        blocked_prefixes = (
            "пользователь:",
            "user:",
            "ассистент:",
            "assistant:",
            "system:",
            "система:"
        )

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            lowered = stripped.casefold()

            if lowered.startswith(blocked_prefixes):
                if cleaned:
                    break

                separator = stripped.find(":")

                if separator >= 0:
                    stripped = stripped[
                        separator + 1:
                    ].strip()

                    if not stripped:
                        continue

            cleaned.append(stripped)

        if not cleaned:
            return ""

        return "\n".join(cleaned).strip()
