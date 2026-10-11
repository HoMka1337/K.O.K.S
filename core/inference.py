import torch
from config import Config

class CodexInference:
    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or Config.DEVICE
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.85,
        top_k: int = 50
    ) -> str:
        if len(prompt) > 600:
            lines = prompt.split('\n')
            if len(lines) > 8:
                prompt = '\n'.join(lines[-8:])
            else:
                prompt = prompt[-600:]
        prompt = prompt + '\n'
        return self.model.generate(
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )