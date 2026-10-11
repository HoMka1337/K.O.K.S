
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
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        prompt = prompt.strip()

        if not prompt:
            return ""

        if not getattr(self.model, "generation_ready", True):
            return ""

        max_new_tokens = max(1, int(max_new_tokens))
        temperature = max(0.01, float(temperature))
        top_k = max(1, int(top_k))

        model_max_len = getattr(self.model, "max_len", None)

        if model_max_len is None:
            model_max_len = getattr(
                self.tokenizer,
                "max_len",
                512
            )

        model_max_len = max(16, int(model_max_len))

        available_tokens = max(
            1,
            model_max_len - max_new_tokens
        )

        if len(prompt) > 600:
            lines = prompt.splitlines()

            if len(lines) > 8:
                prompt = "\n".join(lines[-8:])

            if len(prompt) > 600:
                prompt = prompt[-600:]

        if getattr(self.model, "prompt_style", "agent") == "plain_dialog":
            prompt = prompt.strip() + "\n"
        else:
            prompt = prompt.rstrip() + "\n"

        if hasattr(self.tokenizer, "encode_prompt"):
            token_ids = self.tokenizer.encode_prompt(prompt)
        else:
            token_ids = self.tokenizer.encode(
                prompt,
                add_eos=False
            )

        if len(token_ids) > available_tokens:
            lines = prompt.splitlines()

            while len(lines) > 1:
                lines.pop(0)
                candidate = "\n".join(lines).rstrip() + "\n"

                if hasattr(self.tokenizer, "encode_prompt"):
                    candidate_ids = self.tokenizer.encode_prompt(
                        candidate
                    )
                else:
                    candidate_ids = self.tokenizer.encode(
                        candidate,
                        add_eos=False
                    )

                if len(candidate_ids) <= available_tokens:
                    prompt = candidate
                    token_ids = candidate_ids
                    break

            if len(token_ids) > available_tokens:
                words = prompt.split()

                while len(words) > 1:
                    words.pop(0)
                    candidate = " ".join(words) + "\n"

                    if hasattr(self.tokenizer, "encode_prompt"):
                        candidate_ids = self.tokenizer.encode_prompt(
                            candidate
                        )
                    else:
                        candidate_ids = self.tokenizer.encode(
                            candidate,
                            add_eos=False
                        )

                    if len(candidate_ids) <= available_tokens:
                        prompt = candidate
                        token_ids = candidate_ids
                        break

        max_new_tokens = min(
            max_new_tokens,
            max(1, model_max_len - len(token_ids))
        )

        self.model.eval()

        with torch.inference_mode():
            response = self.model.generate(
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        if not isinstance(response, str):
            return ""

        response = response.strip()

        if not response:
            return ""

        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response
