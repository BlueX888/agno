"""A streamed run must not be replayed after its output has reached the consumer.

The agent retry loop wraps the whole run, so a provider drop in the middle of a
streamed answer used to restart the run: the consumer got a second RunStarted
and a second answer concatenated onto the first, while the final RunOutput
carried only the retry's text. Once content has been streamed, a retry cannot
produce a transcript that matches the RunOutput, so the run must fail instead
of silently duplicating it. A failure before any output is still retried.
"""

from typing import Any, AsyncIterator, Iterator, List

import pytest

from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.run.agent import RunContentEvent, RunErrorEvent, RunOutput, RunOutputEvent, RunStartedEvent
from agno.run.base import RunStatus


class FlakyStreamModel(Model):
    """Fails partway through the first stream, then succeeds on any later attempt."""

    def __init__(self, fail_before_content: bool = False) -> None:
        super().__init__(id="flaky-stream", name="flaky-stream", provider="test")
        self.calls = 0
        self.fail_before_content = fail_before_content

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(content="FULL", role="assistant")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        self.calls += 1
        if self.calls == 1:
            # A mid-stream connection drop: part of the answer reached the consumer.
            if not self.fail_before_content:
                yield ModelResponse(content="PARTIAL", role="assistant")
            raise ModelProviderError(
                message="provider dropped mid-stream", model_name="flaky-stream", model_id="flaky-stream"
            )
        yield ModelResponse(content="FULL", role="assistant")

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        for response in self.invoke_stream(*args, **kwargs):
            yield response

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


async def _run(model: Model, *, async_mode: bool, stream_events: bool, store_events: bool) -> List[RunOutputEvent]:
    agent = Agent(model=model, retries=1, delay_between_retries=0, telemetry=False, store_events=store_events)
    kwargs = {"stream": True, "stream_events": stream_events, "yield_run_output": True}
    if async_mode:
        return [event async for event in agent.arun("hi", **kwargs)]
    return list(agent.run("hi", **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stream_events", [True, False], ids=["events", "no-events"])
@pytest.mark.parametrize("store_events", [True, False], ids=["stored", "unstored"])
async def test_stream_retry_after_partial_output_does_not_replay_the_run(
    async_mode: bool, stream_events: bool, store_events: bool
) -> None:
    model = FlakyStreamModel()

    events = await _run(model, async_mode=async_mode, stream_events=stream_events, store_events=store_events)

    started = [event for event in events if isinstance(event, RunStartedEvent)]
    deltas = [event.content for event in events if isinstance(event, RunContentEvent)]
    errors = [event for event in events if isinstance(event, RunErrorEvent)]
    output = [event for event in events if isinstance(event, RunOutput)][-1]

    # What the consumer received is exactly what the run reports.
    assert "".join(deltas) == output.content == "PARTIAL"
    # One run, one RunStarted (none are emitted when stream_events is off).
    assert len(started) == (1 if stream_events else 0)
    # The partial answer was delivered; replaying the run would append a second answer.
    assert model.calls == 1
    assert output.status == RunStatus.error
    assert errors, "an abandoned streamed run must end with a RunErrorEvent"

    if store_events:
        stored = [getattr(event, "event", None) for event in (output.events or [])]
        assert stored.count("RunStarted") == (1 if stream_events else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stream_events", [True, False], ids=["events", "no-events"])
async def test_stream_retry_before_any_output_is_still_retried(async_mode: bool, stream_events: bool) -> None:
    model = FlakyStreamModel(fail_before_content=True)

    events = await _run(model, async_mode=async_mode, stream_events=stream_events, store_events=False)

    started = [event for event in events if isinstance(event, RunStartedEvent)]
    deltas = [event.content for event in events if isinstance(event, RunContentEvent)]
    errors = [event for event in events if isinstance(event, RunErrorEvent)]
    output = [event for event in events if isinstance(event, RunOutput)][-1]

    # Nothing reached the consumer on the failed attempt, so the retry is safe and useful.
    assert model.calls == 2
    assert len(started) == (1 if stream_events else 0)
    assert "".join(deltas) == output.content == "FULL"
    assert output.status == RunStatus.completed
    assert not errors
