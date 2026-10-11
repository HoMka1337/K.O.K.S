
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from config import Config


class BPETokenizer:
    def __init__(self):
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<SEP>": 4,
            " ": 5,
        }

        self.vocab = self.special_tokens.copy()
        self.inverse_vocab = {
            index: token
            for token, index in self.vocab.items()
        }

        self.merges = []
        self.vocab_size = len(self.vocab)
        self.max_len = 512
        self.format_version = 2

    def __len__(self):
        return self.vocab_size

    def _get_stats(self, word_freqs):
        pairs = Counter()

        for word, freq in word_freqs.items():
            for index in range(len(word) - 1):
                pair = (word[index], word[index + 1])
                pairs[pair] += freq

        return pairs

    def _merge_pair(self, pair, word_freqs):
        result = {}

        for word, freq in word_freqs.items():
            merged = []
            index = 0

            while index < len(word):
                if (
                    index + 1 < len(word)
                    and word[index] == pair[0]
                    and word[index + 1] == pair[1]
                ):
                    merged.append(pair[0] + pair[1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1

            key = tuple(merged)
            result[key] = result.get(key, 0) + freq

        return result

    def train(self, texts: List[str], vocab_size: int = 32000):
        self.vocab = self.special_tokens.copy()
        self.merges = []

        word_freqs = Counter()

        for text in texts:
            if not isinstance(text, str):
                continue

            for word in text.split():
                if word:
                    word_freqs[tuple(word)] += 1

        for word in word_freqs:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        target_size = max(
            len(self.vocab),
            int(vocab_size)
        )

        while len(self.vocab) < target_size:
            stats = self._get_stats(word_freqs)

            if not stats:
                break

            best_pair, frequency = stats.most_common(1)[0]

            if frequency <= 0:
                break

            new_token = best_pair[0] + best_pair[1]

            if new_token in self.vocab:
                break

            word_freqs = self._merge_pair(
                best_pair,
                word_freqs
            )

            self.merges.append(best_pair)
            self.vocab[new_token] = len(self.vocab)

        self.vocab_size = len(self.vocab)

        self.inverse_vocab = {
            index: token
            for token, index in self.vocab.items()
        }

        return self.vocab

    def _encode_word(self, word: str) -> List[int]:
        symbols = list(word)

        for pair in self.merges:
            pair = tuple(pair)
            merged = []
            index = 0

            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == pair[0]
                    and symbols[index + 1] == pair[1]
                ):
                    merged.append(
                        symbols[index] + symbols[index + 1]
                    )
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1

            symbols = merged

        unknown = self.vocab.get("<UNK>", 1)

        return [
            self.vocab.get(symbol, unknown)
            for symbol in symbols
        ]

    def encode(
        self,
        text: str,
        add_eos: bool = True
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        tokens = [
            self.vocab.get("<BOS>", 2)
        ]

        if self.format_version >= 2:
            segments = text.split("\n")
        else:
            segments = [text]

        for segment_index, segment in enumerate(segments):
            if segment_index > 0:
                tokens.append(
                    self.vocab.get("<SEP>", 4)
                )

            words = segment.split()

            for word_index, word in enumerate(words):
                if word_index > 0:
                    tokens.append(
                        self.vocab.get(" ", 5)
                    )

                tokens.extend(
                    self._encode_word(word)
                )

        if add_eos:
            limit = max(2, int(self.max_len))

            tokens = tokens[:limit - 1]
            tokens.append(
                self.vocab.get("<EOS>", 3)
            )

        return tokens

    def encode_prompt(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("Prompt must be a string")

        prompt = text.rstrip("\r\n") + "\n"

        tokens = self.encode(
            prompt,
            add_eos=False
        )

        limit = max(2, int(self.max_len))

        if len(tokens) > limit:
            bos = self.vocab.get("<BOS>", 2)
            tokens = [bos] + tokens[-(limit - 1):]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        result = []

        for token in tokens:
            try:
                token_id = int(token)
            except (TypeError, ValueError):
                continue

            value = self.inverse_vocab.get(
                token_id
            )

            if value is None:
                continue

            if value == "<EOS>":
                break

            if value == "<SEP>":
                if self.format_version >= 2:
                    result.append("\n")
                continue

            if value in (
                "<PAD>",
                "<UNK>",
                "<BOS>"
            ):
                continue

            result.append(value)

        return "".join(result)

    def fingerprint(self):
        payload = {
            "vocab": self.vocab,
            "merges": self.merges,
            "format_version": self.format_version
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":")
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "format_version": self.format_version,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size
        }

        temporary = path.with_name(
            path.name + ".tmp"
        )

        try:
            with temporary.open(
                "w",
                encoding="utf-8"
            ) as stream:
                json.dump(
                    data,
                    stream,
                    ensure_ascii=False,
                    indent=2
                )

            temporary.replace(path)

        finally:
            temporary.unlink(
                missing_ok=True
            )

    def load(self, path: str):
        path = Path(path)

        if not path.is_file():
            return False

        with path.open(
            "r",
            encoding="utf-8"
        ) as stream:
            data = json.load(stream)

        vocab = data.get("vocab")
        merges = data.get("merges")
        special_tokens = data.get("special_tokens")

        if not isinstance(vocab, dict):
            raise ValueError("Invalid tokenizer vocabulary")

        if not isinstance(merges, list):
            raise ValueError("Invalid tokenizer merges")

        if not isinstance(special_tokens, dict):
            raise ValueError("Invalid special tokens")

        vocab = {
            str(token): int(index)
            for token, index in vocab.items()
        }

        special_tokens = {
            str(token): int(index)
            for token, index in special_tokens.items()
        }

        if len(set(vocab.values())) != len(vocab):
            raise ValueError("Duplicate token IDs")

        for token, index in special_tokens.items():
            if vocab.get(token) != index:
                raise ValueError(
                    "Special token mismatch"
                )

        validated_merges = []

        for pair in merges:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(
                    isinstance(item, str)
                    for item in pair
                )
            ):
                raise ValueError(
                    "Invalid BPE merge"
                )

            validated_merges.append(
                tuple(pair)
            )

        self.vocab = vocab
        self.merges = validated_merges
        self.special_tokens = special_tokens

        self.vocab_size = len(vocab)

        self.format_version = int(
            data.get("format_version", 1)
        )

        self.inverse_vocab = {
            index: token
            for token, index in vocab.items()
        }

        return True
