import json
from collections import Counter
from typing import List, Dict, Tuple
from pathlib import Path

from config import Config

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
            ' ': 5,
        }
        self.vocab_size = len(self.special_tokens)
        self.max_len = 512
    
    def __len__(self):
        return self.vocab_size
    
    def _get_stats(self, word_freqs):
        pairs = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] = pairs.get((word[i], word[i+1]), 0) + freq
        return pairs
    
    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = {}
        new_token = pair[0] + pair[1]
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs
    
    def train(self, texts: List[str], vocab_size: int = 32000):
        print("Обучение токенизатора BPE...")
        
        self.vocab = self.special_tokens.copy()
        
        word_freqs = {}
        for text in texts:
            for word in text.split():
                if not word:
                    continue
                char_tuple = tuple(word)
                word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1
        
        for chars in word_freqs.keys():
            for char in chars:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        num_merges = vocab_size - len(self.vocab)
        self.merges = []
        for i in range(num_merges):
            stats = self._get_stats(word_freqs)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Токенизатор обучен. Размер словаря: {self.vocab_size}")
        return self.vocab
    
    def encode(self, text: str) -> List[int]:
        tokens = [self.vocab.get('<BOS>', 2)]
        words = text.split()
        for i, word in enumerate(words):
            if i > 0:
                tokens.append(self.vocab.get(' ', 1))
            tokens.extend(self._encode_word(word))
        tokens.append(self.vocab.get('<EOS>', 3))
        return tokens[:self.max_len]

    def _encode_word(self, word: str) -> List[int]:
        symbols = list(word)
        for pair in self.merges:
            pair = tuple(pair)
            merged = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        unknown = self.vocab.get('<UNK>', 1)
        return [self.vocab.get(symbol, unknown) for symbol in symbols]
    
    def decode(self, tokens: List[int]) -> str:
        result = []
        for token in tokens:
            if token in self.inverse_vocab:
                char = self.inverse_vocab[token]
                if char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']:
                    result.append(char)
        return ''.join(result)
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size,
            }, f, indent=2)
        print(f"Токенизатор сохранён: {path}")
    
    def load(self, path: str):
        if not Path(path).exists():
            print(f"Файл токенизатора не найден: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.special_tokens = data['special_tokens']
            self.vocab_size = data['vocab_size']
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Токенизатор загружен: {path}")