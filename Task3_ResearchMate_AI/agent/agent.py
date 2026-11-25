"""LLM-driven tool-calling agent.

Loop:  query -> LLM decides (answer or tool calls) -> execute tools -> observations
       -> LLM decides again ... -> final answer.

The LLM alone chooses tools and arguments through native function calling
(``bind_tools``); there is no keyword routing anywhere in this code. The loop is
written directly on langchain-core primitives because ``AgentExecutor`` was
removed from LangChain 1.x, which gives us explicit control over iteration
limits, timeouts and the console trace.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent.executor import ToolExecutor
from agent.prompts import build_system_prompt
from config.settings import Settings
from core.exceptions import LLMError
from core.logger import get_logger, redact
from core.models import Source, ToolResult
from core.validators import validate_text

log = get_logger(__name__)

MAX_QUERY_CHARS = 4000
MAX_PARALLEL_CALLS = 4


@dataclass
class TraceEvent:
    """A concise, user-visible step (never the model's private reasoning)."""

    kind: str                 # analyzing | decision | action | observation | warning
    text: str = ""
    tool: str | None = None
    args: dict | None = None
    ok: bool = True


@dataclass
class AgentResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    tool_calls: int = 0
    iterations: int = 0
    limit_reached: bool = False
    error: str | None = None


def build_llm(settings: Settings) -> BaseChatModel:
    """Create the chat model from environment configuration."""
    if not settings.llm_ready:
        raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
        timeout=settings.llm_timeout,
        max_retries=2,
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


class ResearchAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        executor: ToolExecutor,
        settings: Settings,
        on_event: Callable[[TraceEvent], None] | None = None,
    ):
        self.llm = llm
        self.executor = executor
        self.settings = settings
        self.on_event = on_event or (lambda event: None)
        self._llm_with_tools = llm.bind_tools(list(executor.tools.values()))
        self._history: list[BaseMessage] = []

    # ------------------------------------------------------------- memory
    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    def _remember(self, query: str, answer: str) -> None:
        self._history += [HumanMessage(content=query), AIMessage(content=answer)]
        keep = self.settings.history_turns * 2
        self._history = self._history[-keep:] if keep else []

    # --------------------------------------------------------------- core
    def _emit(self, **kwargs) -> None:
        self.on_event(TraceEvent(**kwargs))

    def _call_llm(self, model, messages: list[BaseMessage]) -> AIMessage:
        try:
            return model.invoke(messages)
        except Exception as exc:
            log.warning("LLM call failed: %s", exc)
            raise LLMError(redact(f"{type(exc).__name__}: {exc}")[:300]) from exc

    def run(self, query: str) -> AgentResponse:
        try:
            query = validate_text(query, name="query", max_len=MAX_QUERY_CHARS)
        except Exception as exc:
            return AgentResponse(answer=f"Invalid input: {exc}", error=str(exc))

        self._emit(kind="analyzing", text="Analyzing the query...")
        messages: list[BaseMessage] = [
            SystemMessage(content=build_system_prompt(max_iterations=self.settings.max_iterations)),
            *self._history,
            HumanMessage(content=query),
        ]
        sources: list[Source] = []
        seen_calls: set[str] = set()
        total_calls = 0
        used_tools = False

        try:
            for iteration in range(1, self.settings.max_iterations + 1):
                ai = self._call_llm(self._llm_with_tools, messages)
                messages.append(ai)

                if not ai.tool_calls and not getattr(ai, "invalid_tool_calls", None):
                    if not used_tools:
                        self._emit(kind="decision", text="No tool needed - answering directly.")
                    return self._finish(query, _message_text(ai), sources, total_calls, iteration)

                for position, call in enumerate(ai.tool_calls):
                    total_calls += 1
                    result = self._execute(call, total_calls, seen_calls, over_parallel_cap=position >= MAX_PARALLEL_CALLS)
                    used_tools = True
                    sources.extend(result.sources)
                    messages.append(
                        ToolMessage(content=result.to_llm_text(), tool_call_id=call["id"], name=call["name"])
                    )
                for bad in getattr(ai, "invalid_tool_calls", None) or []:
                    # Malformed JSON arguments: tell the model so it can retry.
                    total_calls += 1
                    self._emit(kind="action", tool=bad.get("name") or "unknown", args={})
                    self._emit(kind="observation", tool=bad.get("name"), ok=False,
                               text="Failed: tool arguments were not valid JSON.")
                    messages.append(ToolMessage(
                        content="ERROR: the tool arguments were not valid JSON. Retry with a valid JSON object.",
                        tool_call_id=bad.get("id") or "invalid", name=bad.get("name") or "unknown",
                    ))

            # Iteration budget exhausted: force a final answer without tools.
            self._emit(kind="warning", text=f"Reached the maximum of {self.settings.max_iterations} iterations.")
            messages.append(SystemMessage(content=(
                "Tool-use limit reached. Do NOT call any more tools. Write the best final answer "
                "you can from the information gathered so far, and state clearly what is missing."
            )))
            final = self._call_llm(self.llm, messages)
            return self._finish(query, _message_text(final), sources, total_calls,
                                self.settings.max_iterations, limit_reached=True)
        except LLMError as exc:
            self._emit(kind="warning", text="The language model request failed.")
            return AgentResponse(
                answer=f"Sorry, I could not get a response from the language model: {exc}",
                sources=self._dedupe(sources), tool_calls=total_calls, error=str(exc),
            )

    # -------------------------------------------------------------- steps
    def _execute(self, call: dict, number: int, seen: set[str], over_parallel_cap: bool = False) -> ToolResult:
        name, args = call["name"], call.get("args", {})
        self._emit(kind="action", tool=name, args=args if isinstance(args, dict) else {})

        key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if number > self.settings.max_tool_calls:
            result = ToolResult.failure(name, "Tool-call budget exhausted. Write your final answer now.")
        elif over_parallel_cap:
            result = ToolResult.failure(
                name, f"Too many tool calls in one step (max {MAX_PARALLEL_CALLS}). Call it again in the next step."
            )
        elif key in seen:
            result = ToolResult.failure(
                name, "Duplicate call: you already made this exact call. Reuse the earlier result or change the arguments."
            )
        else:
            seen.add(key)
            result = self.executor.run(name, args)

        self._emit(kind="observation", tool=name, ok=result.ok, text=result.summary)
        if result.flagged:
            self._emit(kind="warning", text=f"{name} output contained instruction-like text; it is treated as data and ignored.")
        return result

    @staticmethod
    def _dedupe(sources: list[Source]) -> list[Source]:
        seen, unique = set(), []
        for s in sources:
            key = (s.kind, s.url or s.title)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def _finish(self, query, answer, sources, total_calls, iterations, limit_reached=False) -> AgentResponse:
        answer = answer or "I was not able to produce an answer. Please try rephrasing your question."
        self._remember(query, answer)
        return AgentResponse(
            answer=answer, sources=self._dedupe(sources), tool_calls=total_calls,
            iterations=iterations, limit_reached=limit_reached,
        )
