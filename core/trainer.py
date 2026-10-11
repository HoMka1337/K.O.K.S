
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from core.learning import question_key


class CodexDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_len = max(2, int(max_len))
        self.tokens = []

        for text in texts:
            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            tokens = tokenizer.encode(
                text,
                add_eos=True
            )

            tokens = tokens[:self.max_len]

            if len(tokens) < 2:
                continue

            self.tokens.append(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        pad_token = self.tokenizer.vocab.get("<PAD>", 0)

        if len(tokens) < self.max_len:
            tokens = tokens + [
                pad_token
            ] * (self.max_len - len(tokens))

        x = torch.tensor(
            tokens[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            tokens[1:],
            dtype=torch.long
        )

        return x, y


class CodexTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.last_training_metrics = None

    def _make_loader(
        self,
        texts,
        batch_size,
        max_len,
        shuffle=False
    ):
        dataset = CodexDataset(
            texts,
            self.tokenizer,
            max_len=max_len
        )

        if len(dataset) == 0:
            raise ValueError("Dataset contains no usable examples")

        return DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
            num_workers=0
        )

    def _make_scheduler(
        self,
        optimizer,
        total_steps
    ):
        scheduler_name = getattr(
            Config,
            "LR_SCHEDULER",
            None
        )

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(total_steps))
            )

        return None

    def _copy_state(self):
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _calculate_loss(self, logits, targets):
        pad_token = self.tokenizer.vocab.get(
            "<PAD>",
            0
        )

        return F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            targets.reshape(-1),
            ignore_index=pad_token
        )

    def evaluate(self, dataloader):
        if dataloader is None:
            return None

        previous_mode = self.model.training
        self.model.eval()

        pad_token = self.tokenizer.vocab.get(
            "<PAD>",
            0
        )

        total_loss = 0.0
        total_tokens = 0

        try:
            with torch.inference_mode():
                for x, y in dataloader:
                    x = x.to(
                        self.device,
                        non_blocking=self.device.type == "cuda"
                    )

                    y = y.to(
                        self.device,
                        non_blocking=self.device.type == "cuda"
                    )

                    valid_tokens = (
                        y != pad_token
                    ).sum().item()

                    if valid_tokens <= 0:
                        continue

                    logits = self.model(x)

                    loss = F.cross_entropy(
                        logits.reshape(
                            -1,
                            logits.size(-1)
                        ).float(),
                        y.reshape(-1),
                        ignore_index=pad_token,
                        reduction="sum"
                    )

                    if not torch.isfinite(loss).item():
                        return float("inf")

                    total_loss += loss.item()
                    total_tokens += valid_tokens

            if total_tokens == 0:
                return float("inf")

            return total_loss / total_tokens

        finally:
            self.model.train(previous_mode)

    def train(
        self,
        texts: List[str],
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 3e-4,
        max_len: int = 256,
        save_path: str = "models/KoKs_model.pt",
        validation_split: float = 0.1,
        validation_texts=None
    ):
        epochs = int(epochs)
        batch_size = int(batch_size)
        max_len = int(max_len)
        lr = float(lr)

        if epochs < 1:
            raise ValueError("Epochs must be positive")

        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive")

        if not texts:
            raise ValueError("Training data is empty")

        if not validation_texts:
            raise ValueError(
                "An independent validation set is required"
            )

        train_examples = [
            text.strip()
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        validation_examples = [
            text.strip()
            for text in validation_texts
            if isinstance(text, str) and text.strip()
        ]

        if not train_examples:
            raise ValueError("Training data is empty")

        if not validation_examples:
            raise ValueError("Validation data is empty")

        validation_keys = {
            question_key(text)
            for text in validation_examples
        }

        train_examples = [
            text
            for text in train_examples
            if question_key(text) not in validation_keys
        ]

        if not train_examples:
            raise ValueError(
                "No training examples remain after validation filtering"
            )

        max_len = min(
            max_len,
            int(self.model.max_len)
        )

        if max_len < 2:
            raise ValueError(
                "Maximum sequence length must be at least 2"
            )

        train_loader = self._make_loader(
            train_examples,
            batch_size,
            max_len,
            shuffle=True
        )

        validation_loader = self._make_loader(
            validation_examples,
            batch_size,
            max_len,
            shuffle=False
        )

        original_mode = self.model.training
        original_state = self._copy_state()

        self.model.to(self.device)

        baseline_loss = self.evaluate(
            validation_loader
        )

        if not math.isfinite(baseline_loss):
            raise ValueError(
                "Baseline validation loss is not finite"
            )

        self.last_training_metrics = {
            "baseline_loss": baseline_loss,
            "best_loss": baseline_loss,
            "best_epoch": 0,
            "epochs": [],
            "accepted": False,
            "train_examples": len(train_loader.dataset),
            "validation_examples": len(
                validation_loader.dataset
            )
        }

        amp_enabled = (
            self.device.type == "cuda"
            and bool(getattr(Config, "USE_AMP", False))
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(
                getattr(Config, "WEIGHT_DECAY", 0.01)
            )
        )

        scheduler = self._make_scheduler(
            optimizer,
            len(train_loader) * epochs
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled
        )

        grad_clip = float(
            getattr(Config, "GRAD_CLIP", 1.0)
        )

        patience = int(
            getattr(
                Config,
                "EARLY_STOPPING_PATIENCE",
                0
            )
        )

        empty_cache_every = int(
            getattr(
                Config,
                "EMPTY_CACHE_EVERY_N_STEPS",
                0
            )
        )

        best_loss = baseline_loss
        best_state = None
        stale_epochs = 0
        global_step = 0
        accepted = False

        try:
            for epoch in range(epochs):
                self.model.train()

                total_train_loss = 0.0
                train_steps = 0

                progress = tqdm(
                    train_loader,
                    desc=f"Эпоха {epoch + 1}/{epochs}"
                )

                for x, y in progress:
                    x = x.to(
                        self.device,
                        non_blocking=self.device.type == "cuda"
                    )

                    y = y.to(
                        self.device,
                        non_blocking=self.device.type == "cuda"
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=amp_enabled
                    ):
                        logits = self.model(x)

                        loss = self._calculate_loss(
                            logits.float(),
                            y
                        )

                    if not torch.isfinite(loss).item():
                        raise ValueError(
                            "Training loss is not finite"
                        )

                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)

                    if grad_clip > 0:
                        gradient_norm = (
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                grad_clip
                            )
                        )

                        if not torch.isfinite(
                            gradient_norm
                        ).item():
                            raise ValueError(
                                "Gradient norm is not finite"
                            )

                    scaler.step(optimizer)
                    scaler.update()

                    if scheduler is not None:
                        scheduler.step()

                    loss_value = loss.detach().item()

                    total_train_loss += loss_value
                    train_steps += 1
                    global_step += 1

                    progress.set_postfix(
                        loss=f"{loss_value:.4f}"
                    )

                    if (
                        self.device.type == "cuda"
                        and empty_cache_every > 0
                        and global_step % empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()

                if train_steps == 0:
                    raise ValueError(
                        "No training batches were processed"
                    )

                average_train_loss = (
                    total_train_loss / train_steps
                )

                validation_loss = self.evaluate(
                    validation_loader
                )

                self.last_training_metrics["epochs"].append({
                    "epoch": epoch + 1,
                    "train_loss": average_train_loss,
                    "validation_loss": validation_loss
                })

                print(
                    f"Эпоха {epoch + 1}: "
                    f"loss = {average_train_loss:.4f}, "
                    f"val_loss = {validation_loss:.4f}"
                )

                if (
                    math.isfinite(validation_loss)
                    and validation_loss < best_loss
                ):
                    best_loss = validation_loss
                    best_state = self._copy_state()
                    stale_epochs = 0

                    self.last_training_metrics[
                        "best_loss"
                    ] = best_loss

                    self.last_training_metrics[
                        "best_epoch"
                    ] = epoch + 1

                else:
                    stale_epochs += 1

                if (
                    patience > 0
                    and stale_epochs >= patience
                ):
                    break

            if best_state is None:
                return False

            self.model.load_state_dict(
                best_state
            )

            self.last_training_metrics[
                "accepted"
            ] = True

            self.save(save_path)
            accepted = True

            return True

        finally:
            if not accepted:
                self.model.load_state_dict(
                    original_state
                )

                self.last_training_metrics[
                    "accepted"
                ] = False

            self.model.train(original_mode)

            del original_state
            del best_state

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def save(self, path: str):
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fd, temporary_name = tempfile.mkstemp(
            prefix=".checkpoint_",
            suffix=".tmp",
            dir=str(destination.parent)
        )

        os.close(fd)

        temporary = Path(temporary_name)

        try:
            self._save_checkpoint(
                temporary
            )

            if destination.is_file():
                previous = destination.with_name(
                    destination.name + ".previous"
                )

                shutil.copy2(
                    destination,
                    previous
                )

            os.replace(
                temporary,
                destination
            )

        finally:
            temporary.unlink(
                missing_ok=True
            )

    def _save_checkpoint(self, path):
        layers = getattr(
            self.model,
            "layers",
            []
        )

        if len(layers) > 0:
            n_heads = layers[0].attention.n_heads
            d_ff = layers[0].ff.fc1.out_features
        else:
            n_heads = 4
            d_ff = 1024

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "prompt_style": getattr(
                self.model,
                "prompt_style",
                "agent"
            ),
            "generation_ready": getattr(
                self.model,
                "generation_ready",
                True
            ),
            "training_metrics": self.last_training_metrics,
            "tokenizer_format": self.tokenizer.format_version,
            "tokenizer_fingerprint": self.tokenizer.fingerprint(),
            "vocab_size": self.model.vocab_size,
            "d_model": self.model.d_model,
            "max_len": self.model.max_len,
            "config": {
                "n_heads": n_heads,
                "n_layers": len(layers),
                "d_ff": d_ff
            }
        }

        torch.save(
            checkpoint,
            path
        )

    def load(self, path: str):
        checkpoint_path = Path(path)

        if not checkpoint_path.is_file():
            return None

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid model checkpoint"
            )

        state = checkpoint.get(
            "model_state_dict"
        )

        if not isinstance(state, dict):
            raise ValueError(
                "Checkpoint does not contain model weights"
            )

        expected_fingerprint = (
            self.tokenizer.fingerprint()
        )

        checkpoint_fingerprint = checkpoint.get(
            "tokenizer_fingerprint"
        )

        if (
            checkpoint_fingerprint is not None
            and checkpoint_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Checkpoint and tokenizer do not match"
            )

        checkpoint_format = checkpoint.get(
            "tokenizer_format",
            1
        )

        if (
            checkpoint_format
            != self.tokenizer.format_version
        ):
            raise ValueError(
                "Tokenizer format mismatch"
            )

        checkpoint_vocab_size = checkpoint.get(
            "vocab_size",
            len(self.tokenizer)
        )

        if (
            checkpoint_vocab_size
            != len(self.tokenizer)
        ):
            raise ValueError(
                "Tokenizer vocabulary size mismatch"
            )

        self.model.load_state_dict(
            state,
            strict=True
        )

        self.model.to(self.device)

        self.model.prompt_style = checkpoint.get(
            "prompt_style",
            "agent"
        )

        self.model.generation_ready = bool(
            checkpoint.get(
                "generation_ready",
                True
            )
        )

        self.last_training_metrics = checkpoint.get(
            "training_metrics"
        )

        self.model.eval()

        return checkpoint


def read_checkpoint_shape(path: str):
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        return None

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        TypeError
    ):
        return None

    if not isinstance(checkpoint, dict):
        return None

    state = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(state, dict):
        return None

    shape = {}

    embedding = state.get(
        "token_embedding.weight"
    )

    if isinstance(embedding, torch.Tensor):
        if embedding.ndim != 2:
            return None

        shape["vocab_size"] = int(
            embedding.shape[0]
        )

        shape["d_model"] = int(
            embedding.shape[1]
        )

    else:
        shape["vocab_size"] = checkpoint.get(
            "vocab_size"
        )

        shape["d_model"] = checkpoint.get(
            "d_model"
        )

    positions = state.get(
        "pos_encoding.pe"
    )

    if isinstance(positions, torch.Tensor):
        if positions.ndim != 2:
            return None

        shape["max_len"] = int(
            positions.shape[0]
        )

        if shape.get("d_model") is None:
            shape["d_model"] = int(
                positions.shape[1]
            )

    else:
        shape["max_len"] = checkpoint.get(
            "max_len"
        )

    n_layers = 0

    for key in state:
        if not isinstance(key, str):
            continue

        if (
            key.startswith("layers.")
            and key.endswith(
                ".attention.wq.weight"
            )
        ):
            parts = key.split(".")

            if len(parts) < 4:
                continue

            try:
                index = int(parts[1])
            except ValueError:
                continue

            n_layers = max(
                n_layers,
                index + 1
            )

    shape["n_layers"] = n_layers

    first_ff = state.get(
        "layers.0.ff.fc1.weight"
    )

    if isinstance(first_ff, torch.Tensor):
        shape["d_ff"] = int(
            first_ff.shape[0]
        )

    config = checkpoint.get(
        "config",
        {}
    )

    if not isinstance(config, dict):
        config = {}

    if "d_ff" not in shape:
        shape["d_ff"] = config.get(
            "d_ff",
            1024
        )

    shape["n_heads"] = config.get(
        "n_heads",
        8
    )

    required = (
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "max_len"
    )

    for key in required:
        value = shape.get(key)

        if value is None:
            return None

        try:
            value = int(value)
        except (TypeError, ValueError):
            return None

        if value <= 0:
            return None

        shape[key] = value

    if shape["d_model"] % shape["n_heads"] != 0:
        return None

    return shape
