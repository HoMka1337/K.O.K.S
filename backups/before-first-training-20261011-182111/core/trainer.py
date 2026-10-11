import math
import os
import shutil
import tempfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import List

from config import Config


class CodexDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokens = []
        for text in tqdm(texts, desc="Токенизация"):
            tokens = tokenizer.encode(text)[:max_len]
            self.tokens.append(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        pad_token = self.tokenizer.vocab.get("<PAD>", 0)
        if len(tokens) < self.max_len:
            tokens = tokens + [pad_token] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class CodexTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        print(f"Устройство: {device}")
        print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    def train(
        self,
        texts: List[str],
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 3e-4,
        max_len: int = 256,
        save_path: str = "models/KoKs_model.pt",
        validation_split: float = 0.1,
        validation_texts=None,
    ):
        if not texts or epochs < 1:
            raise ValueError("Training requires nonempty data and positive epochs")
        if not validation_texts:
            raise ValueError("A fixed, independent validation set is required")
        from core.learning import question_key
        held_out = {question_key(t) for t in validation_texts}
        texts = [t for t in texts if question_key(t) not in held_out]
        if not texts:
            raise ValueError("No training examples remain after excluding validation questions")
        max_len = min(max_len, self.model.max_len)
        if max_len < 2:
            raise ValueError("max_len must be >= 2")
        dataloader = DataLoader(CodexDataset(texts, self.tokenizer, max_len),
                                batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(CodexDataset(validation_texts, self.tokenizer, max_len),
                                       batch_size=batch_size)
        original = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        original_mode = self.model.training
        baseline = self.evaluate(validation_loader)
        if not math.isfinite(baseline):
            raise ValueError("Non-finite baseline validation loss")
        best_state = None
        best_loss = baseline
        amp_enabled = Config.USE_AMP and str(self.device).startswith("cuda")
        accepted = False
        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
            )
            scheduler = self._make_scheduler(optimizer, len(dataloader) * epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

            self.model.train()
            global_step = 0

            for epoch in range(epochs):
                total_loss = 0
                progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{epochs}")
                for x, y in progress_bar:
                    x = x.to(self.device, non_blocking=Config.PIN_MEMORY)
                    y = y.to(self.device, non_blocking=Config.PIN_MEMORY)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(
                        enabled=amp_enabled,
                        dtype=torch.bfloat16 if Config.AMP_DTYPE == "bf16" else torch.float16,
                    ):
                        logits = self.model(x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            y.reshape(-1),
                            ignore_index=self.tokenizer.vocab.get("<PAD>", 0),
                        )
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()

                    total_loss += loss.item()
                    global_step += 1
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                    if Config.EMPTY_CACHE_EVERY_N_STEPS > 0 and global_step % Config.EMPTY_CACHE_EVERY_N_STEPS == 0:
                        if self.device == "cuda":
                            torch.cuda.empty_cache()

                avg_loss = total_loss / max(1, len(dataloader))
                validation_loss = self.evaluate(validation_loader)
                metric = validation_loss if validation_loss is not None else avg_loss
                suffix = f", val_loss = {validation_loss:.4f}" if validation_loss is not None else ""
                print(f"Эпоха {epoch+1}: loss = {avg_loss:.4f}{suffix}")

                if math.isfinite(metric) and metric < best_loss:
                    best_loss = metric
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}

            if best_state is None:
                print("Candidate rejected: validation did not improve")
                return False
            self.model.load_state_dict(best_state)
            self.save(save_path)
            accepted = True
            print(f"Candidate accepted: {baseline:.4f} -> {best_loss:.4f}")
            return True
        finally:
            if not accepted:
                self.model.load_state_dict(original)
            self.model.train(original_mode)

    def _make_scheduler(self, optimizer, total_steps):
        if Config.LR_SCHEDULER == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        return None

    def evaluate(self, dataloader):
        if dataloader is None:
            return None
        was_training = self.model.training
        self.model.eval()
        total_loss, count = 0.0, 0
        pad = self.tokenizer.vocab.get("<PAD>", 0)
        try:
            with torch.no_grad():
                for x, y in dataloader:
                    logits = self.model(x.to(self.device))
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           y.to(self.device).reshape(-1),
                                           ignore_index=pad, reduction="sum")
                    total_loss += loss.item()
                    count += int((y != pad).sum())
            return total_loss / count if count else float("inf")
        finally:
            self.model.train(was_training)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=Path(path).parent, suffix=".tmp")
        os.close(fd)
        try:
            self._save_checkpoint(temporary)
            if Path(path).exists():
                shutil.copy2(path, str(path) + ".previous")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "tokenizer_format": self.tokenizer.format_version,
                "tokenizer_fingerprint": self.tokenizer.fingerprint(),
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "max_len": self.model.max_len,
                "config": {
                    "n_heads": self.model.layers[0].attention.n_heads
                    if self.model.layers else 4,
                    "n_layers": len(self.model.layers),
                    "d_ff": self.model.layers[0].ff.fc1.out_features
                    if self.model.layers else 1024,
                },
            },
            path,
        )
        print(f"Модель сохранена: {path}")

    def load(self, path: str):
        if not Path(path).exists():
            print(f"Файл модели не найден: {path}")
            return None
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("tokenizer_fingerprint") not in (None, self.tokenizer.fingerprint()):
            raise ValueError("Checkpoint and tokenizer do not match")
        if checkpoint.get("tokenizer_format", 1) != self.tokenizer.format_version:
            raise ValueError("Checkpoint and tokenizer formats do not match")
        if checkpoint.get("vocab_size", len(self.tokenizer)) != len(self.tokenizer):
            raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Модель загружена из {path}")
        return checkpoint


def read_checkpoint_shape(path: str):
    if not Path(path).exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"Не удалось прочитать чекпоинт: {error}")
        return None

    state = checkpoint.get("model_state_dict")
    if state is None:
        return None

    shape = {}
    shape["vocab_size"] = checkpoint.get("vocab_size")
    if shape["vocab_size"] is None and "token_embedding.weight" in state:
        shape["vocab_size"] = state["token_embedding.weight"].shape[0]

    if "token_embedding.weight" in state:
        shape["d_model"] = state["token_embedding.weight"].shape[1]
    if "pos_encoding.pe" in state:
        shape["max_len"] = state["pos_encoding.pe"].shape[0]
        if "d_model" not in shape:
            shape["d_model"] = state["pos_encoding.pe"].shape[1]

    n_layers = 0
    for key in state.keys():
        if key.startswith("layers.") and key.endswith(".attention.wq.weight"):
            try:
                idx = int(key.split(".")[1])
                n_layers = max(n_layers, idx + 1)
            except (ValueError, IndexError):
                continue
    shape["n_layers"] = n_layers

    if "layers.0.attention.wq.weight" in state:
        shape["d_model"] = state["layers.0.attention.wq.weight"].shape[0]

    if "layers.0.ff.fc1.weight" in state:
        shape["d_ff"] = state["layers.0.ff.fc1.weight"].shape[0]

    cfg = checkpoint.get("config") or {}
    shape["n_heads"] = cfg.get("n_heads", 8)

    return shape