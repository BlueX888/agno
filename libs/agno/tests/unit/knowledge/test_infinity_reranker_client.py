"""InfinityReranker must keep its cached client usable across rerank calls.

``infinity_client`` is not part of the test extras (it pins httpx<0.28.0, which
conflicts with the version agno requires), so the module is stubbed here. The
stub mirrors the generated client's contract: ``Client.__enter__``/``__exit__``
delegate to the wrapped ``httpx.Client``, so exiting closes the underlying
client and re-entering it raises ``RuntimeError``. With that behaviour a cached
client that is entered per rerank call only works once.
"""

import importlib
import sys
import types
from typing import Any, List

import httpx
import pytest

from agno.knowledge.document import Document

RERANK_RESPONSE = {
    "results": [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
    ],
    "model": "test-model",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


class FakeClient:
    """Minimal stand-in for ``infinity_client.Client``.

    The real wrapper closes the underlying httpx client on ``__exit__`` while
    staying cached on the reranker, which is what breaks the second call.
    """

    def __init__(self, base_url: str = "", token: str = "", verify_ssl: bool = True, **kwargs: Any):
        self.base_url = base_url
        self.token = token
        self.verify_ssl = verify_ssl
        self._client = None
        self._async_client = None

    def get_httpx_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url)
        return self._client

    def set_httpx_client(self, client: httpx.Client) -> None:
        self._client = client

    def get_async_httpx_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(base_url=self.base_url)
        return self._async_client

    def set_async_httpx_client(self, client: httpx.AsyncClient) -> None:
        self._async_client = client

    def __enter__(self) -> "FakeClient":
        self.get_httpx_client().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.get_httpx_client().__exit__(*args)

    async def __aenter__(self) -> "FakeClient":
        await self.get_async_httpx_client().__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.get_async_httpx_client().__aexit__(*args)


class FakeAuthenticatedClient(FakeClient):
    pass


class FakeRerankInput:
    def __init__(self, data: dict):
        self.data = data

    @classmethod
    def from_dict(cls, data: dict) -> "FakeRerankInput":
        return cls(data)

    def to_dict(self) -> dict:
        return self.data


def _parse_response(payload: dict) -> Any:
    return types.SimpleNamespace(
        results=[
            types.SimpleNamespace(index=item["index"], relevance_score=item["relevance_score"])
            for item in payload["results"]
        ]
    )


class FakeRerank:
    @staticmethod
    def sync(client: FakeClient, body: FakeRerankInput) -> Any:
        response = client.get_httpx_client().post("/rerank", json=body.to_dict())
        response.raise_for_status()
        return _parse_response(response.json())

    @staticmethod
    async def asyncio(client: FakeClient, body: FakeRerankInput) -> Any:
        response = await client.get_async_httpx_client().post("/rerank", json=body.to_dict())
        response.raise_for_status()
        return _parse_response(response.json())


@pytest.fixture
def infinity_module(monkeypatch: pytest.MonkeyPatch):
    """Import agno's InfinityReranker against the stub package above."""
    package = types.ModuleType("infinity_client")
    api = types.ModuleType("infinity_client.api")
    default = types.ModuleType("infinity_client.api.default")
    models = types.ModuleType("infinity_client.models")

    package.Client = FakeClient
    package.AuthenticatedClient = FakeAuthenticatedClient
    default.rerank = FakeRerank
    models.RerankInput = FakeRerankInput

    monkeypatch.setitem(sys.modules, "infinity_client", package)
    monkeypatch.setitem(sys.modules, "infinity_client.api", api)
    monkeypatch.setitem(sys.modules, "infinity_client.api.default", default)
    monkeypatch.setitem(sys.modules, "infinity_client.models", models)
    monkeypatch.delitem(sys.modules, "agno.knowledge.reranker.infinity", raising=False)

    module = importlib.import_module("agno.knowledge.reranker.infinity")
    yield module
    monkeypatch.delitem(sys.modules, "agno.knowledge.reranker.infinity", raising=False)


def _make_reranker(module: Any, requests: List[httpx.Request]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=RERANK_RESPONSE)

    transport = httpx.MockTransport(handler)
    wrapper = module.Client(base_url="http://localhost:7997")
    wrapper.set_httpx_client(httpx.Client(base_url="http://localhost:7997", transport=transport))
    wrapper.set_async_httpx_client(httpx.AsyncClient(base_url="http://localhost:7997", transport=transport))

    reranker = module.InfinityReranker()
    reranker._client = wrapper
    return reranker


def _contents_and_scores(documents: List[Document]) -> List[Any]:
    return [(doc.content, doc.reranking_score) for doc in documents]


def test_rerank_works_on_every_call_with_the_cached_client(infinity_module):
    requests: List[httpx.Request] = []
    reranker = _make_reranker(infinity_module, requests)

    first = reranker.rerank(query="q", documents=[Document(content="a"), Document(content="b")])
    assert _contents_and_scores(first) == [("b", 0.9), ("a", 0.1)]

    second = reranker.rerank(query="q", documents=[Document(content="a"), Document(content="b")])
    assert _contents_and_scores(second) == [("b", 0.9), ("a", 0.1)]

    assert len(requests) == 2


async def test_arerank_works_on_every_call_with_the_cached_client(infinity_module):
    requests: List[httpx.Request] = []
    reranker = _make_reranker(infinity_module, requests)

    first = await reranker.arerank(query="q", documents=[Document(content="a"), Document(content="b")])
    assert _contents_and_scores(first) == [("b", 0.9), ("a", 0.1)]

    second = await reranker.arerank(query="q", documents=[Document(content="a"), Document(content="b")])
    assert _contents_and_scores(second) == [("b", 0.9), ("a", 0.1)]

    assert len(requests) == 2
