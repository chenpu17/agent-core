# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for TOOL_CALL_ERROR dedup between _ToolMeta and AbilityManager.

The tool lifecycle wrapper (_ToolMeta in foundation/tool/base.py) fires
TOOL_CALL_ERROR for exceptions escaping ``tool.invoke``. AbilityManager's
result-processing loop fires the same event again for every exception it
receives, so a failed tool call used to be reported twice — and in parallel
mode the second trigger runs outside the lane task, where request-scoped
contextvars are invisible, so consumers keyed on those see a ghost result.

These tests pin the single-report contract:
- exactly one TOOL_CALL_ERROR per failed tool call (sequential and parallel);
- no TOOL_CALL_ERROR for ToolInterruptException (control flow, not a failure);
- pre-invoke failures (malformed arguments) still report exactly once — they
  never pass through the lifecycle wrapper;
- a TimeoutError raised by the tool itself is not double-reported (the
  anyio fail-after scope raising TimeoutError is a different, unmarked path).
"""

from __future__ import annotations

import asyncio

from openjiuwen.core.foundation.llm import ToolCall
from openjiuwen.core.foundation.tool import LocalFunction, ToolCard
from openjiuwen.core.runner import Runner
from openjiuwen.core.runner.callback.events import ToolCallEvents
from openjiuwen.core.single_agent.ability_manager import AbilityManager
from openjiuwen.core.single_agent.interrupt.exception import ToolInterruptException
from openjiuwen.core.single_agent.interrupt.response import InterruptRequest
from openjiuwen.core.single_agent.rail.base import (
    AgentCallbackContext,
    ToolCallInputs,
)


class _FakeCallbackManager:
    async def execute(self, event, ctx) -> None:
        return None


class _FakeAgent:
    def __init__(self) -> None:
        self.agent_callback_manager = _FakeCallbackManager()


def _make_tool(name: str, behavior) -> LocalFunction:
    card = ToolCard(id=name, name=name, description=f"{name} desc", stateless=True)

    async def _func(**_):
        behavior()

    return LocalFunction(card=card, func=_func)


def _ctx(agent: _FakeAgent, tool_name: str) -> AgentCallbackContext:
    return AgentCallbackContext(
        agent=agent,
        inputs=ToolCallInputs(tool_name=tool_name, tool_call=None, tool_args="{}"),
        config=None, session=None, context=None, extra={},
    )


def _call(name: str, arguments: str = "{}") -> ToolCall:
    return ToolCall(id="c1", type="function", name=name, arguments=arguments)


async def _collect_error_events(tool: LocalFunction, *, parallel: bool,
                                arguments: str = "{}") -> list:
    """Run one call through AbilityManager.execute; return TOOL_CALL_ERROR payloads."""
    events: list = []

    async def spy(**kw):
        events.append(kw)

    await Runner.start()
    fw = Runner.callback_framework
    fw.register_sync(ToolCallEvents.TOOL_CALL_ERROR, spy)
    am = AbilityManager(owner_id="lifecycle-dedup")
    try:
        am.add_ability(tool.card, tool)
        await am.execute(_ctx(_FakeAgent(), tool.card.name),
                         [_call(tool.card.name, arguments=arguments)],
                         None, parallel_tool_calls=parallel)
    finally:
        am.remove_ability(tool.card.name)
        fw.unregister_sync(ToolCallEvents.TOOL_CALL_ERROR, spy)
        await Runner.stop()
    return events


def test_failed_tool_call_reports_single_error_event_both_modes() -> None:
    async def _run():
        for parallel in (False, True):
            def raise_boom():
                raise ValueError("boom")

            tool = _make_tool("boom", raise_boom)
            events = await _collect_error_events(tool, parallel=parallel)
            assert len(events) == 1, f"parallel={parallel}: {events}"
            assert events[0]["tool_name"] == "boom"

    asyncio.run(_run())


def test_tool_interrupt_is_not_an_error_event() -> None:
    async def _run():
        def raise_interrupt():
            raise ToolInterruptException(InterruptRequest(message="need user"))

        tool = _make_tool("intr", raise_interrupt)
        events = await _collect_error_events(tool, parallel=False)
        assert events == []

    asyncio.run(_run())


def test_malformed_arguments_still_report_error_event() -> None:
    async def _run():
        tool = _make_tool("ok", lambda: None)
        events = await _collect_error_events(tool, parallel=False,
                                             arguments="{not json")
        assert len(events) == 1
        assert events[0]["tool_name"] == "ok"

    asyncio.run(_run())


def test_tool_raised_timeout_not_double_reported() -> None:
    async def _run():
        def raise_timeout():
            raise TimeoutError("tool's own timeout")

        tool = _make_tool("slow", raise_timeout)
        events = await _collect_error_events(tool, parallel=False)
        assert len(events) == 1, f"{events}"

    asyncio.run(_run())
