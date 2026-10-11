"""Regression tests: the output model must not corrupt pause/stop transcripts.

When an agent has an ``output_model`` and the run pauses for tool confirmation
(or breaks on ``stop_after_tool_call``), the transcript's last message is NOT
the main model's draft answer: it is the assistant message carrying the pending
tool_call (pause path) or the tool result (stop path). The output-model helper
used to unconditionally pop that last message from the run's own transcript and
the output model then appended its fabricated answer to the same list, so:

- the PAUSED run was persisted with no record of the pending tool call, its
  content was a fabricated final answer produced before the tool ever ran, and
  on continue the executed tool's result was appended with no preceding
  assistant tool_call — the provider payload carried an orphaned ``tool``
  message, which OpenAI/Anthropic/Gemini reject with 400;
- the ``stop_after_tool_call`` variant popped the tool result, so the output
  model was called with an orphaned assistant tool_call as the last message.

The helper must only drop the last message when it is actually the main model's
draft answer (a plain assistant message), and the output model must not run
while a run is pausing for tool confirmation.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from agno.agent.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.base import Model
from agno.models.message import Message, MessageMetrics
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput, ToolExecution
from agno.run.base import RunStatus
from agno.run.requirement import RunRequirement
from agno.tools.function import Function


class ToolThenTextModel(Model):
    """Offline model: first invoke emits a get_weather tool call, later invokes return plain text."""

    def __init__(self):
        super().__init__(id="main", name="main", provider="test")
        self.received: List[List[Any]] = []
        self.calls = 0

    def get_instructions_for_model(self, *args, **kwargs):
        return None

    def get_system_message_for_model(self, *args, **kwargs):
        return None

    async def aget_instructions_for_model(self, *args, **kwargs):
        return None

    async def aget_system_message_for_model(self, *args, **kwargs):
        return None

    def parse_args(self, *args, **kwargs):
        return {}

    def invoke(self, *args, messages=None, **kwargs) -> ModelResponse:
        self.received.append(list(messages or []))
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="",
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
                response_usage=MessageMetrics(),
            )
        return ModelResponse(content="summarized answer", role="assistant", response_usage=MessageMetrics())

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        yield self.invoke(*args, **kwargs)

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        yield await self.ainvoke(*args, **kwargs)

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return ModelResponse(content="x", role="assistant")

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return ModelResponse(content="x", role="assistant")


class PlainTextModel(Model):
    """Offline output model: always answers with a fixed text."""

    def __init__(self, text: str):
        super().__init__(id="output", name="output", provider="test")
        self.text = text
        self.received: List[List[Any]] = []

    def get_instructions_for_model(self, *args, **kwargs):
        return None

    def get_system_message_for_model(self, *args, **kwargs):
        return None

    async def aget_instructions_for_model(self, *args, **kwargs):
        return None

    async def aget_system_message_for_model(self, *args, **kwargs):
        return None

    def parse_args(self, *args, **kwargs):
        return {}

    def invoke(self, *args, messages=None, **kwargs) -> ModelResponse:
        self.received.append(list(messages or []))
        return ModelResponse(content=self.text, role="assistant", response_usage=MessageMetrics())

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        yield self.invoke(*args, **kwargs)

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        yield await self.ainvoke(*args, **kwargs)

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return ModelResponse(content=self.text, role="assistant")

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return ModelResponse(content=self.text, role="assistant")


def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


def _weather_tool():
    return Function.from_callable(get_weather)


def _orphaned_tool_call_ids(messages) -> List[str]:
    """Tool-call ids in a payload that would be rejected by providers.

    An orphan is either a ``tool`` message with no preceding assistant
    tool_call announcing it, or an assistant tool_call with no following
    ``tool`` result (e.g. an assistant tool_call left as the last message).
    """
    announced: set = set()
    answered: set = set()
    problems: List[str] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    announced.add(tc_id)
        elif m.role == "tool" and m.tool_call_id:
            if m.tool_call_id not in announced:
                problems.append(m.tool_call_id)
            answered.add(m.tool_call_id)
    for tc_id in announced - answered:
        problems.append(tc_id)
    return problems


def _pending_tool_call_message(run_output: RunOutput) -> Optional[Message]:
    for m in run_output.messages or []:
        if m.role == "assistant" and m.tool_calls:
            return m
    return None


class TestOutputModelPauseAndContinue:
    def test_paused_run_keeps_pending_tool_call_and_continue_payload_is_valid(self):
        main_model = ToolThenTextModel()
        output_model = PlainTextModel("FINAL ANSWER (tool never ran!)")
        weather_fn = _weather_tool()
        weather_fn.requires_confirmation = True
        agent = Agent(model=main_model, output_model=output_model, tools=[weather_fn], db=InMemoryDb())

        run1: RunOutput = agent.run(input="weather in Paris?")
        assert run1.status == RunStatus.paused

        # The paused run is persisted with the pending tool call still on record:
        # the transcript must keep the assistant message that carries the tool_call.
        pending = _pending_tool_call_message(run1)
        assert pending is not None, "paused run transcript lost the assistant message carrying the pending tool_call"

        # The output model must not fabricate an answer before the tool ever ran.
        assert output_model.received == [], "output model ran while the run was pausing for tool confirmation"

        # Continue by confirming the tool: the executed tool's result is appended
        # to the same transcript, so the post-continue model payload must pair
        # every tool message with the assistant tool_call that announced it.
        confirmed = ToolExecution(
            tool_call_id=run1.tools[0].tool_call_id,
            tool_name="get_weather",
            tool_args={"city": "Paris"},
            requires_confirmation=True,
            confirmed=True,
        )
        run2 = agent.continue_run(run_id=run1.run_id, requirements=[RunRequirement(tool_execution=confirmed)])
        assert run2.status == RunStatus.completed

        post_continue_payload = main_model.received[-1]
        orphans = _orphaned_tool_call_ids(post_continue_payload)
        assert orphans == [], (
            f"post-continue model payload has orphaned tool messages {orphans} — providers reject with 400"
        )

    async def test_async_paused_run_keeps_pending_tool_call_and_continue_payload_is_valid(self):
        main_model = ToolThenTextModel()
        output_model = PlainTextModel("FINAL ANSWER (tool never ran!)")
        weather_fn = _weather_tool()
        weather_fn.requires_confirmation = True
        agent = Agent(model=main_model, output_model=output_model, tools=[weather_fn], db=InMemoryDb())

        run1: RunOutput = await agent.arun(input="weather in Paris?")
        assert run1.status == RunStatus.paused

        pending = _pending_tool_call_message(run1)
        assert pending is not None, "paused run transcript lost the assistant message carrying the pending tool_call"
        assert output_model.received == [], "output model ran while the run was pausing for tool confirmation"

        confirmed = ToolExecution(
            tool_call_id=run1.tools[0].tool_call_id,
            tool_name="get_weather",
            tool_args={"city": "Paris"},
            requires_confirmation=True,
            confirmed=True,
        )
        run2 = await agent.acontinue_run(run_id=run1.run_id, requirements=[RunRequirement(tool_execution=confirmed)])
        assert run2.status == RunStatus.completed

        post_continue_payload = main_model.received[-1]
        orphans = _orphaned_tool_call_ids(post_continue_payload)
        assert orphans == [], (
            f"post-continue model payload has orphaned tool messages {orphans} — providers reject with 400"
        )


class TestOutputModelStopAfterToolCall:
    def test_stop_after_tool_call_keeps_tool_result_in_transcript_and_output_model_payload(self):
        main_model = ToolThenTextModel()
        output_model = PlainTextModel("final!")
        weather_fn = _weather_tool()
        weather_fn.stop_after_tool_call = True
        agent = Agent(model=main_model, output_model=output_model, tools=[weather_fn], db=InMemoryDb())

        run1: RunOutput = agent.run(input="weather in Paris?")
        assert run1.status == RunStatus.completed

        # The tool result must survive in the persisted transcript, and the
        # output model's answer must still land as the final assistant message.
        tool_results = [m for m in run1.messages or [] if m.role == "tool"]
        assert len(tool_results) == 1, "persisted transcript lost the executed tool's result message"
        assert run1.content == run1.messages[-1].content, "output model answer should be the final transcript message"

        # The output model must see the tool result, and its payload must not end
        # on an orphaned assistant tool_call.
        output_payload = output_model.received[0]
        payload_tool_results = [m for m in output_payload if m.role == "tool"]
        assert len(payload_tool_results) == 1, "output model payload lost the executed tool's result message"
        assert _orphaned_tool_call_ids(output_payload) == []


class TestOutputModelNormalRun:
    """The plain output-model flow keeps its documented transcript contract."""

    def test_output_model_answer_is_the_final_transcript_message(self):
        main_model = ToolThenTextModel()
        main_model.calls = 1  # skip the tool-call turn: every invoke returns plain text
        output_model = PlainTextModel("final answer")
        agent = Agent(model=main_model, output_model=output_model, db=InMemoryDb())

        run1: RunOutput = agent.run(input="weather in Paris?")
        assert run1.status == RunStatus.completed
        assert run1.content == "final answer"

        assert output_model.received, "output model should be called on a normal run"

        # Exactly one assistant message: the main model's draft is replaced by
        # the output model's answer, and that answer is the last transcript message.
        assistant_messages = [m for m in run1.messages or [] if m.role == "assistant"]
        assert len(assistant_messages) == 1
        assert run1.content == run1.messages[-1].content


class TestGetMessagesForOutputModelDraftHandling:
    """Direct coverage of the helper: drop only a trailing plain assistant draft."""

    def _agent(self) -> Agent:
        return Agent(name="output-model-agent", output_model=PlainTextModel("final!"))

    def test_drops_trailing_assistant_draft(self):
        from agno.agent._messages import get_messages_for_output_model

        agent = self._agent()
        messages = [
            Message(role="user", content="weather in Paris?"),
            Message(role="assistant", content="draft answer"),
        ]

        result = get_messages_for_output_model(agent, messages)

        assert [m.content for m in result] == ["weather in Paris?"], "draft assistant answer should be dropped"

    def test_trailing_assistant_tool_call_is_kept(self):
        from agno.agent._messages import get_messages_for_output_model

        agent = self._agent()
        messages = [
            Message(role="user", content="weather in Paris?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            ),
        ]

        result = get_messages_for_output_model(agent, messages)

        assert [m.role for m in result] == ["user", "assistant"], (
            "an assistant message carrying a pending tool_call is not a draft answer and must be kept"
        )

    def test_trailing_tool_result_is_kept(self):
        from agno.agent._messages import get_messages_for_output_model

        agent = self._agent()
        messages = [
            Message(role="user", content="weather in Paris?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            ),
            Message(role="tool", content="The weather in Paris is sunny.", tool_call_id="call_1"),
        ]

        result = get_messages_for_output_model(agent, messages)

        assert [m.role for m in result] == ["user", "assistant", "tool"], (
            "a trailing tool result is not the main model's draft answer and must be kept"
        )
