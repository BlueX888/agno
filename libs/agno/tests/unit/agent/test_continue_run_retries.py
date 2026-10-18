"""Retries on the async continue-run path must not replay the continuation setup.

``_acontinue_run`` / ``_acontinue_run_stream`` run the whole continuation
pipeline inside their ``for attempt in range(num_attempts)`` loop, including
the fork/truncate step and the append of the caller's ``input``. That
preparation mutates the in-memory ``run_response`` carried across attempts, so
a retry re-forks the previous attempt's fork and appends the same user
instruction a second time. The sync ``continue_run_dispatch`` performs that
preparation once, before ``_continue_run``'s retry loop, so only the async
executors are affected.

These tests pin the single-append, single-fork behavior for the public
``acontinue_run`` forms: ``run_response=`` and ``run_id=``, streaming and
non-streaming.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

import pytest

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.run.base import RunStatus


class FailOnceModel(Model):
    """Offline model: answers ANSWER; raises once when ``fail_next`` is set."""

    def __init__(self) -> None:
        super().__init__(id="fail-once", name="fail-once", provider="test")
        self.fail_next = False
        self.user_messages_seen: list[list[Any]] = []

    def _answer(self, messages: Any) -> ModelResponse:
        self.user_messages_seen.append([m.content for m in messages if m.role == "user"])
        return ModelResponse(content="ANSWER", role="assistant")

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("transient provider failure")
        return self._answer(kwargs.get("messages") or [])

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self.invoke(*args, **kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self.invoke(*args, **kwargs)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


async def _continue(agent: Agent, first: Any, *, stream: bool, use_run_id: bool) -> Any:
    kwargs: dict[str, Any] = {"input": "NEW INSTRUCTION", "stream": stream}
    if use_run_id:
        kwargs.update(run_id=first.run_id, session_id=first.session_id)
    else:
        kwargs["run_response"] = first
    if not stream:
        return await agent.acontinue_run(**kwargs)

    final = None
    async for event in agent.acontinue_run(**kwargs, yield_run_output=True):
        if getattr(event, "messages", None) is not None:
            final = event
    return final


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("use_run_id", [False, True], ids=["run-response", "run-id"])
async def test_acontinue_run_input_not_duplicated_on_retry(stream: bool, use_run_id: bool):
    model = FailOnceModel()
    agent = Agent(
        model=model,
        db=InMemoryDb() if use_run_id else None,
        retries=1,
        delay_between_retries=0,
        telemetry=False,
    )
    first = agent.run("ORIGINAL QUESTION")

    model.fail_next = True
    cont = await _continue(agent, first, stream=stream, use_run_id=use_run_id)

    assert cont.status == RunStatus.completed
    # The failed attempt must not append its own copy of the caller's input.
    assert [m.content for m in cont.messages if m.role == "user"] == [
        "ORIGINAL QUESTION",
        "NEW INSTRUCTION",
    ]
    # The retried model call must see one copy of the instruction.
    assert model.user_messages_seen[-1] == ["ORIGINAL QUESTION", "NEW INSTRUCTION"]
    # The retry must not fork the previous attempt's ephemeral fork: lineage
    # points at the run the caller continued.
    assert cont.forked_from_run_id == first.run_id
