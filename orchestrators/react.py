"""ReAct orchestrator - reasoning and acting in a loop."""

import json
import logging
from typing import Any, Dict, List, TYPE_CHECKING

from benchmark.mcp_client import MCPClient
from benchmark.llm_client import LLMClient
from benchmark.models import BenchmarkConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from .base import AgentOrchestrator
from .claude_code import price_rates, compute_cache_aware_cost

logger = logging.getLogger(__name__)


class ReactOrchestrator(AgentOrchestrator):

    async def execute(self) -> Dict[str, Any]:
        """Execute the ReAct (Reason-Act-Observe) loop until the LLM stops
        calling tools or max_iterations is reached."""
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=self.config.user_prompt),
        ]

        conversation_flow = [
            {"type": "system_message", "content": self.config.system_prompt},
            {"type": "user_message", "content": self.config.user_prompt},
        ]
        tools_used = []
        tool_results = []

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} ---")

            # Reason: ask the LLM what to do next
            response = await self.llm_client.invoke_with_tools(
                messages, self.available_tools
            )
            reasoning_details = (getattr(response, "additional_kwargs", None) or {}).get(
                "reasoning_details"
            )
            if reasoning_details:
                logger.debug(f"Preserving reasoning_details for next turn ({len(reasoning_details)} items)")

            messages.append(response)

            usage_metadata = (
                response.usage_metadata if hasattr(response, "usage_metadata") else {}
            )
            response_metadata = (
                response.response_metadata
                if hasattr(response, "response_metadata")
                else {}
            )

            conversation_flow.append(
                {
                    "type": "ai_message",
                    "content": response.content,
                    "usage_metadata": usage_metadata,
                    "response_metadata": response_metadata,
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in (response.tool_calls or [])
                    ],
                }
            )

            logger.info(f"LLM Response: {response.content}")

            # Terminate if the LLM decided no further tool calls are needed
            if not response.tool_calls or len(response.tool_calls) == 0:
                logger.info("No tool calls requested. Task complete.")
                break

            # Act + Observe: execute each tool call and feed results back
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.debug(f"Tool arguments: {tool_args}")

                exec_result = await self._execute_tool_call(tool_name, tool_args)
                tool_result = exec_result["result"]
                target_gym = exec_result["gym_server"]

                logger.info(f"Tool result success: {tool_result.get('success')}")

                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                        "gym_server": target_gym,
                    }
                )

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result.get("result", {})),
                        tool_call_id=tool_call.get("id", ""),
                    )
                )

                conversation_flow.append(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": tool_result,
                        "gym_server": target_gym,
                    }
                )

        self._conversation_flow = conversation_flow
        return {
            "final_response": messages[-1].content if messages else "",
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }

    def get_result_metadata(self) -> Dict[str, Any]:
        """Cache-aware cost + token totals for the run, from per-turn usage_metadata.

        LangChain's usage_metadata.input_tokens is the FULL input (incl. cached); the
        cache split lives in input_token_details. Convert to the Anthropic bucket shape
        (uncached input, cache_read, cache_creation) so cache reads bill at the cached
        rate — same accounting as the claude_code arm.
        """
        cf = getattr(self, "_conversation_flow", []) or []
        in_total = out_total = cache_read = cache_creation = 0
        for e in cf:
            if e.get("type") != "ai_message":
                continue
            u = e.get("usage_metadata") or {}
            in_total += u.get("input_tokens", 0) or 0
            out_total += u.get("output_tokens", 0) or 0
            d = u.get("input_token_details") or {}
            cache_read += d.get("cache_read", 0) or 0
            cache_creation += d.get("cache_creation", 0) or 0
        usage = {
            "input_tokens": max(in_total - cache_read - cache_creation, 0),
            "output_tokens": out_total,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }
        return compute_cache_aware_cost(usage, price_rates(getattr(self.llm_client, "model", None)))
