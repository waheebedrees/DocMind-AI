from collections.abc import Iterator


class EmbeddingTextSplitter:
    def __init__(
        self,
        tokenizer,
        *,
        content_budget: int,
        overlap_tokens: int,
    ) -> None:
        if content_budget <= 0:
            raise ValueError("max_tokens must be positive")

        if not 0 <= overlap_tokens < content_budget:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")

        self._tokenizer = tokenizer
        self.max_tokens = content_budget
        self.overlap_tokens = overlap_tokens

    def encode_ids(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.encode_ids(text))

    def count_model_tokens(self, text: str) -> int:
        return len(
            self.encode_ids(
                text,
                add_special_tokens=True,
            )
        )

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def fit(self, text: str) -> str:
        token_ids = self.encode_ids(text)

        if len(token_ids) <= self.max_tokens:
            return text

        size = self.max_tokens

        while size > 0:
            candidate = self.decode_ids(token_ids[:size])

            if self.count_tokens(candidate) <= self.max_tokens:
                return candidate

            size -= 1

        return ""

    def split(
        self,
        text: str,
        *,
        prefix: str = "",
    ) -> Iterator[str]:
        prefix_ids = self.encode_ids(prefix) if prefix else []

        if len(prefix_ids) >= self.max_tokens:
            raise ValueError(f"Context prefix has {len(prefix_ids)} tokens, but the available limit is {self.max_tokens}")

        body_ids = self.encode_ids(text)
        body_limit = self.max_tokens - len(prefix_ids)

        if not body_ids:
            fitted = self.fit(prefix)

            if fitted:
                yield fitted

            return

        step = max(
            1,
            body_limit - self.overlap_tokens,
        )

        start = 0

        while start < len(body_ids):
            token_window = body_ids[start : start + body_limit]

            piece = prefix + self.decode_ids(token_window)
            fitted_piece = self.fit(piece)

            if fitted_piece:
                yield fitted_piece

            if start + body_limit >= len(body_ids):
                break

            start += step
