from typing import Protocol

import httpx


class Reranker(Protocol):
    async def score(self, query: str, texts: list[str]) -> list[float]: ...


class TEIReranker:
    """HuggingFace text-embeddings-inference /rerank endpoint
    (e.g. BAAI/bge-reranker-v2-m3). Swap in a Cohere or Jina client behind the same Protocol."""

    def __init__(self, client: httpx.AsyncClient, url: str):
        self._client = client
        self._url = url

    async def score(self, query: str, texts: list[str]) -> list[float]:
        resp = await self._client.post(
            self._url,
            json={"query": query, "texts": texts, "truncate": True, "raw_scores": False},
        )
        resp.raise_for_status()
        scores = [0.0] * len(texts)
        for item in resp.json():
            scores[item["index"]] = float(item["score"])
        return scores
