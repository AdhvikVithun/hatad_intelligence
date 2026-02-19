"""Ollama LLM client for local model inference.

Supports:
  - Structured outputs via JSON Schema (format: { schema })
  - Chain-of-thought reasoning (think: true → message.thinking)
  - Tool/function calling with multi-round execution loop
"""

import json
import time
import random
import asyncio
import httpx
from typing import Callable, Awaitable
import logging
from app.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT, LLM_MAX_RETRIES,
    LLM_CONTEXT_WINDOW, LLM_MAX_INPUT_CHARS, LLM_WARN_INPUT_CHARS,
    LLM_USE_STRUCTURED_OUTPUTS, LLM_USE_THINKING, LLM_USE_TOOLS,
    LLM_TOOL_CALL_MAX_ROUNDS,
    VISION_MODEL, VISION_TIMEOUT, VISION_CONTEXT_WINDOW,
)

logger = logging.getLogger(__name__)

# Type for progress callback: async fn(stage, message, details_dict)
LLMProgressCallback = Callable[[str, str, dict], Awaitable[None]]

# No-op callback default
async def _noop_cb(stage: str, message: str, details: dict) -> None:
    pass

# Token budget for finalize/revalidation passes.  These produce only a JSON
# result object (no chain-of-thought), so 16K tokens is more than enough.
# Using the full context window here causes prompt+predict to overshoot num_ctx,
# leaving zero tokens for output → Ollama returns empty content.
FINALIZE_PREDICT_BUDGET = 16384


def _build_finalize_hint(schema: dict | None) -> str:
    """Build a dynamic JSON structure hint from the target schema.

    Instead of hard-coding only `checks` + `group_score_deduction`, this
    inspects the schema's required/properties so the finalize prompt matches
    the actual output contract (e.g. Pass 1 also needs `chain_of_title`,
    `active_encumbrances`, etc.).
    """
    if not schema or not isinstance(schema, dict):
        return (
            '{"checks": [{"rule_code": "...", "rule_name": "...", '
            '"severity": "CRITICAL|HIGH|MEDIUM|LOW", "status": "PASS|FAIL|WARNING", '
            '"explanation": "...", "recommendation": "...", "evidence": "..."}], '
            '"group_score_deduction": <number>}'
        )

    props = schema.get("properties", {})
    required = schema.get("required", list(props.keys()))
    parts = []
    for key in required:
        prop = props.get(key, {})
        ptype = prop.get("type", "string")
        if ptype == "array":
            item_type = prop.get("items", {}).get("type", "object")
            if key == "checks":
                parts.append(
                    '"checks": [{"rule_code": "...", "rule_name": "...", '
                    '"severity": "CRITICAL|HIGH|MEDIUM|LOW", "status": "PASS|FAIL|WARNING", '
                    '"explanation": "...", "recommendation": "...", "evidence": "..."}]'
                )
            elif item_type == "object":
                item_props = prop.get("items", {}).get("properties", {})
                inner = ", ".join(f'"{k}": "..."' for k in list(item_props.keys())[:4])
                parts.append(f'"{key}": [{{{inner}}}]')
            else:
                parts.append(f'"{key}": ["..."]')
        elif ptype == "integer":
            parts.append(f'"{key}": <number>')
        elif ptype == "string":
            parts.append(f'"{key}": "..."')
        elif ptype == "boolean":
            parts.append(f'"{key}": true|false')
        else:
            parts.append(f'"{key}": "..."')
    return "{" + ", ".join(parts) + "}"


async def call_llm(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.1,
    expect_json: bool | dict = True,
    task_label: str = "",
    on_progress: LLMProgressCallback | None = None,
    think: bool = False,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
) -> dict | str:
    """Call local Ollama model and return response.
    
    Args:
        prompt: User prompt text
        system_prompt: System prompt for role/context
        temperature: LLM temperature (low = deterministic)
        expect_json: True for basic JSON mode, or a JSON Schema dict for structured outputs.
                     False for free-text response.
        task_label: Human-readable label for this LLM call
        on_progress: Async callback for progress updates
        think: If True, enable chain-of-thought (model returns thinking + content)
        tools: List of Ollama tool definitions for function calling
    
    Returns:
        Parsed JSON dict (with optional _thinking key) or raw string
    """
    cb = on_progress or _noop_cb
    # ── Build messages ──
    messages = []
    if system_prompt:
        sp = system_prompt
        # When thinking is disabled, explicitly instruct the model not to reason
        # (some models ignore the API think:false flag and think anyway)
        if not think:
            sp += "\n\nIMPORTANT: Do NOT include any chain-of-thought, reasoning, or analysis. Output ONLY the requested content directly."
        messages.append({"role": "system", "content": sp})
    messages.append({"role": "user", "content": prompt})

    # ── Input size safety ──
    total_input_chars = sum(len(m["content"]) for m in messages)
    if total_input_chars > LLM_MAX_INPUT_CHARS:
        excess = total_input_chars - LLM_MAX_INPUT_CHARS
        original_len = len(prompt)
        truncated = prompt[: original_len - excess - 300]
        truncated += (
            f"\n\n[... INPUT TRUNCATED for context safety — "
            f"original {original_len:,} chars, kept {len(truncated):,} chars ...]"
        )
        messages[-1]["content"] = truncated
        logger.warning(
            f"[{task_label or 'LLM'}] Input truncated: {original_len:,} → {len(truncated):,} chars"
        )
    elif total_input_chars > LLM_WARN_INPUT_CHARS:
        logger.warning(
            f"[{task_label or 'LLM'}] Large input: {total_input_chars:,} chars (~{total_input_chars // 4:,} tokens)"
        )

    # Calculate token estimate (rough: 1 token ≈ 4 chars)
    prompt_chars = sum(len(m["content"]) for m in messages)
    prompt_tokens_est = prompt_chars // 4

    label = task_label or "LLM Call"

    # ── Determine format parameter ──
    # NOTE: Schema-enforced format + tools + think is an unstable triple combo
    # in Ollama — the model often returns empty content after tool rounds.
    # When tools are active, we downgrade to basic JSON mode for tool rounds,
    # but store the original schema for a schema-enforced finalize pass.
    use_tools = tools if (tools and LLM_USE_TOOLS) else None
    original_schema = None  # Stash the full schema for post-tool finalize
    if expect_json:
        if isinstance(expect_json, dict) and LLM_USE_STRUCTURED_OUTPUTS:
            if use_tools:
                # Downgrade during tool rounds, but remember the schema
                format_param = "json"
                schema_enforced = False
                original_schema = expect_json  # Will be used in finalize
            else:
                format_param = expect_json  # JSON Schema dict (no tools → safe)
                schema_enforced = True
        else:
            format_param = "json"
            schema_enforced = False
    else:
        format_param = ""
        schema_enforced = False

    # ── Determine think parameter ──
    use_think = think and LLM_USE_THINKING

    # Keep a copy of the original messages for clean retries
    original_messages = [m.copy() for m in messages]

    await cb("llm_start", f"{label}", {
        "type": "llm_start",
        "task": label,
        "model": "HATAD AI \u2726 Reasoning",
        "prompt_chars": prompt_chars,
        "prompt_tokens_est": prompt_tokens_est,
        "temperature": temperature,
        "expect_json": bool(expect_json),
        "schema_enforced": schema_enforced,
        "thinking_enabled": use_think,
        "tools_enabled": bool(use_tools),
    })

    last_error = None
    content = ""
    thinking_text = ""
    for attempt in range(LLM_MAX_RETRIES):
        t0 = time.time()

        if attempt > 0:
            await cb("llm_retry", f"{label} — Retry {attempt}/{LLM_MAX_RETRIES}", {
                "type": "llm_retry",
                "task": label,
                "attempt": attempt + 1,
                "reason": str(last_error),
            })

        await cb("llm_waiting", f"{label} — Sending to HATAD AI Reasoning...", {
            "type": "llm_waiting",
            "task": label,
            "model": "HATAD AI ✦ Reasoning",
            "attempt": attempt + 1,
        })

        try:
            # Exponential backoff with jitter on retries
            if attempt > 0:
                backoff = min(2 ** attempt + random.uniform(0, 1), 30)
                await asyncio.sleep(backoff)

            # Dynamic token budget:
            # 1. Explicit max_tokens from caller → task-specific budget
            # 2. think=true → full context window (CoT can consume 25K+ tokens)
            # 3. think=false → 49152 (model template may still think)
            if max_tokens:
                predict_budget = max_tokens
            elif use_think:
                predict_budget = LLM_CONTEXT_WINDOW
            else:
                predict_budget = 49152

            body = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": predict_budget,
                    "num_ctx": LLM_CONTEXT_WINDOW,
                },
                "format": format_param,
                "think": use_think,  # Always explicit — prevents model-level default thinking
            }

            # Tool calling
            if use_tools:
                body["tools"] = use_tools

            # Per-call client — each concurrent task gets its own client
            # This avoids "client has been closed" errors with shared clients
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=body,
                )
                response.raise_for_status()
                elapsed = time.time() - t0
                result = response.json()
                message = result["message"]
                content = message.get("content", "")

                # ── Extract thinking (chain-of-thought) ──
                thinking_text = message.get("thinking", "")
                if thinking_text:
                    await cb("llm_thinking", f"{label} — CoT reasoning ({len(thinking_text):,} chars)", {
                        "type": "llm_thinking",
                        "task": label,
                        "thinking_chars": len(thinking_text),
                        "thinking_preview": thinking_text[:300],
                    })

                # ── Handle tool calls ──
                tool_calls = message.get("tool_calls")
                tool_call_count = 0
                tools_used = set()
                if tool_calls and use_tools:
                    from app.pipeline.tools import execute_tool
                    
                    # Append assistant message with tool_calls
                    messages.append(message)
                    
                    tool_round = 0
                    while tool_calls and tool_round < LLM_TOOL_CALL_MAX_ROUNDS:
                        tool_round += 1
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "unknown")
                            fn_args = fn.get("arguments", {})
                            tool_call_count += 1
                            tools_used.add(fn_name)
                            
                            await cb("llm_tool_call", f"{label} — Calling {fn_name}()", {
                                "type": "llm_tool_call",
                                "task": label,
                                "tool_name": fn_name,
                                "tool_args": fn_args,
                                "round": tool_round,
                            })
                            
                            # Run sync tool executor in a thread so the event
                            # loop stays free — needed for search_documents()
                            # which calls async embedding via asyncio.run().
                            # Use copy_context() so ContextVar values (rag_store,
                            # embed_fn, memory_bank) are visible in the thread.
                            import contextvars
                            ctx = contextvars.copy_context()
                            tool_result = await asyncio.get_running_loop().run_in_executor(
                                None, ctx.run, execute_tool, fn_name, fn_args
                            )
                            
                            await cb("llm_tool_result", f"{label} — {fn_name} returned", {
                                "type": "llm_tool_result",
                                "task": label,
                                "tool_name": fn_name,
                                "result_preview": tool_result[:200],
                                "round": tool_round,
                            })
                            
                            # Send tool result back to model
                            messages.append({
                                "role": "tool",
                                "content": tool_result,
                            })
                        
                        # Re-call the model with tool results
                        body["messages"] = messages
                        response = await client.post(
                            f"{OLLAMA_BASE_URL}/api/chat",
                            json=body,
                        )
                        response.raise_for_status()
                        result = response.json()
                        message = result["message"]
                        content = message.get("content", "")
                        tool_calls = message.get("tool_calls")
                        
                        # Capture any additional thinking
                        extra_thinking = message.get("thinking", "")
                        if extra_thinking:
                            thinking_text += "\n\n" + extra_thinking
                        
                        if tool_calls:
                            messages.append(message)
                    
                    # Update elapsed time after tool rounds
                    elapsed = time.time() - t0

                    # ── Finalize step: if content is STILL empty after tools,
                    # re-call without tools/thinking to force JSON output ──
                    # Now uses the original JSON Schema if one was provided,
                    # reclaiming structured output enforcement.
                    if not content or not content.strip():
                        logger.info(f"[{label}] Content empty after {tool_call_count} tool call(s), sending finalize request (basic JSON mode)")
                        await cb("llm_finalize", f"{label} — Finalizing (generating JSON output)...", {
                            "type": "llm_finalize",
                            "task": label,
                            "tool_calls_completed": tool_call_count,
                            "schema_enforced": False,
                        })

                        # ── Summarize tool history to slim down the prompt ──
                        # Instead of replaying every tool-call/tool-result message
                        # (which can add 10-30K tokens), collapse them into one
                        # concise summary so the finalize request fits comfortably.
                        tool_summary_parts = []
                        trimmed_messages = []
                        for m in messages:
                            role = m.get("role", "")
                            if role == "tool":
                                # Extract tool result and truncate to 500 chars
                                snippet = m.get("content", "")[:500]
                                tool_summary_parts.append(snippet)
                            elif role == "assistant" and m.get("tool_calls"):
                                # Record which tools were called
                                for tc in m["tool_calls"]:
                                    fn = tc.get("function", {})
                                    tool_summary_parts.append(f"Called {fn.get('name', '?')}({json.dumps(fn.get('arguments', {}))[:200]})")
                            else:
                                trimmed_messages.append(m)

                        if tool_summary_parts:
                            trimmed_messages.append({
                                "role": "user",
                                "content": (
                                    "Here is a summary of the tool results you gathered:\n"
                                    + "\n---\n".join(tool_summary_parts)
                                ),
                            })

                        schema_hint = _build_finalize_hint(original_schema)
                        trimmed_messages.append({
                            "role": "user",
                            "content": (
                                "You have finished calling tools. Now produce your FINAL answer as a valid JSON object. "
                                f"The JSON MUST have this structure: {schema_hint}. "
                                "Include ALL the check results based on the tool outputs you received. "
                                "IMPORTANT: You must output EVERY check for this verification group — "
                                "do not stop after 1-2 checks. Provide the complete set of checks "
                                "with detailed evidence and explanations for each one. "
                                "Return ONLY the JSON object, no explanation or commentary."
                            ),
                        })

                        # Always use basic JSON mode for finalize — schema-enforced
                        # constrained decoding with complex schemas causes extreme
                        # slowdown (0.3 tok/s) and truncated output. We validate
                        # required fields via the post-tool schema check instead.
                        finalize_format = "json"

                        finalize_body = {
                            "model": OLLAMA_MODEL,
                            "messages": trimmed_messages,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": FINALIZE_PREDICT_BUDGET,
                                "num_ctx": LLM_CONTEXT_WINDOW,
                            },
                            "format": finalize_format,
                            "think": False,
                        }

                        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as fin_client:
                            fin_response = await fin_client.post(
                                f"{OLLAMA_BASE_URL}/api/chat",
                                json=finalize_body,
                            )
                            fin_response.raise_for_status()
                            fin_result = fin_response.json()
                            fin_content = fin_result["message"].get("content", "")
                            if fin_content and fin_content.strip():
                                content = fin_content
                                result = fin_result  # Update metrics source for accurate tok/s
                                schema_enforced = False  # basic JSON mode, not schema-enforced
                                logger.info(f"[{label}] Finalize produced {len(content)} chars")
                            else:
                                logger.warning(f"[{label}] Finalize returned empty, falling back to basic JSON")
                                # Fallback: retry finalize with basic JSON mode
                                finalize_body["format"] = "json"
                                async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as fb_fin_client:
                                    fb_fin_response = await fb_fin_client.post(
                                        f"{OLLAMA_BASE_URL}/api/chat",
                                        json=finalize_body,
                                    )
                                    fb_fin_response.raise_for_status()
                                    fb_fin_result = fb_fin_response.json()
                                    fb_fin_content = fb_fin_result["message"].get("content", "")
                                    if fb_fin_content and fb_fin_content.strip():
                                        content = fb_fin_content
                                        result = fb_fin_result  # Update metrics source
                                        logger.info(f"[{label}] Basic JSON finalize produced {len(content)} chars")
                                    else:
                                        logger.warning(f"[{label}] Both finalize attempts returned empty")
                        elapsed = time.time() - t0

                    # ── Post-tool schema enforcement pass ──
                    # If we got content with basic JSON but had an original schema,
                    # validate and re-request with schema enforcement if malformed
                    elif original_schema and content and content.strip():
                        try:
                            preliminary = json.loads(content)
                            # Quick schema check: verify required top-level keys exist
                            schema_required = original_schema.get("required", [])
                            missing_keys = [k for k in schema_required if k not in preliminary]
                            if missing_keys:
                                logger.info(f"[{label}] Post-tool content missing schema keys {missing_keys}, doing schema-enforced re-request")
                                await cb("llm_finalize", f"{label} — Schema re-validation pass...", {
                                    "type": "llm_finalize",
                                    "task": label,
                                    "schema_revalidation": True,
                                    "missing_keys": missing_keys,
                                })
                                messages.append(message)
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "Your JSON response is missing required fields: "
                                        f"{', '.join(missing_keys)}. Please output the complete JSON "
                                        "with ALL required fields. Return ONLY valid JSON."
                                    ),
                                })
                                reval_body = {
                                    "model": OLLAMA_MODEL,
                                    "messages": messages,
                                    "stream": False,
                                    "options": {
                                        "temperature": temperature,
                                        "num_predict": FINALIZE_PREDICT_BUDGET,
                                        "num_ctx": LLM_CONTEXT_WINDOW,
                                    },
                                    "format": "json",  # Basic JSON — full schema causes 0.3 tok/s
                                    "think": False,
                                }
                                async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as rv_client:
                                    rv_response = await rv_client.post(
                                        f"{OLLAMA_BASE_URL}/api/chat",
                                        json=reval_body,
                                    )
                                    rv_response.raise_for_status()
                                    rv_result = rv_response.json()
                                    rv_content = rv_result["message"].get("content", "")
                                    if rv_content and rv_content.strip():
                                        content = rv_content
                                        schema_enforced = True
                                        logger.info(f"[{label}] Schema re-validation produced {len(content)} chars")
                        except (json.JSONDecodeError, Exception) as schema_err:
                            logger.info(f"[{label}] Post-tool schema check skipped: {schema_err}")

                # ── Generic fallback: if content is STILL empty (no tools called,
                # or tool-finalize also failed), try one last call without
                # tools or thinking to force the model to just produce JSON ──
                if expect_json and (not content or not content.strip()):
                    logger.info(f"[{label}] Content still empty, attempting no-tools/no-think finalize")
                    await cb("llm_finalize", f"{label} — Fallback finalize (no tools, no thinking)...", {
                        "type": "llm_finalize",
                        "task": label,
                        "fallback": True,
                    })

                    # Build a simplified message list: system + user + retry nudge
                    fallback_schema_hint = _build_finalize_hint(original_schema)
                    fallback_messages = [m.copy() for m in original_messages]
                    fallback_messages.append({
                        "role": "user",
                        "content": (
                            "Return ONLY a valid JSON object with all the check results. "
                            f"Use this exact structure: {fallback_schema_hint}. "
                            "Do not call any tools. Do not explain. Just output the JSON."
                        ),
                    })

                    fallback_body = {
                        "model": OLLAMA_MODEL,
                        "messages": fallback_messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": FINALIZE_PREDICT_BUDGET,
                            "num_ctx": LLM_CONTEXT_WINDOW,
                        },
                        "format": "json",
                        "think": False,
                    }

                    try:
                        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as fb_client:
                            fb_response = await fb_client.post(
                                f"{OLLAMA_BASE_URL}/api/chat",
                                json=fallback_body,
                            )
                            fb_response.raise_for_status()
                            fb_result = fb_response.json()
                            fb_content = fb_result["message"].get("content", "")
                            if fb_content and fb_content.strip():
                                content = fb_content
                                result = fb_result  # Update metrics source
                                logger.info(f"[{label}] Fallback finalize produced {len(content)} chars")
                            else:
                                logger.warning(f"[{label}] Fallback finalize also returned empty")
                    except Exception as fb_err:
                        logger.warning(f"[{label}] Fallback finalize error: {fb_err}")

                    elapsed = time.time() - t0

                # Extract Ollama metrics
                resp_tokens = result.get("eval_count", len(content) // 4)
                prompt_tokens = result.get("prompt_eval_count", prompt_tokens_est)
                total_duration = result.get("total_duration", 0) / 1e9  # nanoseconds → seconds
                # Use Ollama's eval_duration for accurate generation speed
                # (excludes prompt processing and tool call wait time)
                eval_duration = result.get("eval_duration", 0) / 1e9
                if eval_duration > 0:
                    tokens_per_sec = resp_tokens / eval_duration
                else:
                    tokens_per_sec = resp_tokens / elapsed if elapsed > 0 else 0

                await cb("llm_response", f"{label} — {resp_tokens} tokens in {elapsed:.1f}s ({tokens_per_sec:.1f} tok/s)", {
                    "type": "llm_response",
                    "task": label,
                    "model": "HATAD AI ✦ Reasoning",
                    "elapsed_seconds": round(elapsed, 2),
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": resp_tokens,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "response_chars": len(content),
                    "ollama_duration": round(total_duration, 2),
                })

                if expect_json:
                    await cb("llm_parsing", f"{label} — Parsing JSON response...", {
                        "type": "llm_parsing",
                        "task": label,
                        "schema_enforced": schema_enforced,
                    })

                    # Try multiple sources for JSON extraction:
                    # 1. Content directly
                    # 2. If content is empty or has no JSON markers, try thinking
                    # 3. If both fail, try concatenated content + thinking
                    parsed = None
                    parse_error = None

                    # Attempt 1: Parse content directly
                    if content and content.strip():
                        try:
                            parsed = _parse_json_response(content)
                        except (json.JSONDecodeError, ValueError) as e:
                            parse_error = e
                            logger.info(f"[{label}] Content ({len(content)} chars) failed JSON parse: {e}")

                    # Attempt 2: If content is empty or had no JSON, try thinking
                    if parsed is None and thinking_text:
                        content_has_json = content and ('{' in content or '[' in content)
                        if not content_has_json:
                            logger.info(f"[{label}] Trying JSON extraction from thinking ({len(thinking_text):,} chars)")
                            try:
                                parsed = _parse_json_response(thinking_text)
                            except (json.JSONDecodeError, ValueError) as e2:
                                logger.info(f"[{label}] Thinking also failed JSON parse: {e2}")

                    # Attempt 3: Concatenate both and scan
                    if parsed is None and thinking_text and content:
                        combined = thinking_text + "\n" + content
                        if '{' in combined or '[' in combined:
                            logger.info(f"[{label}] Trying JSON extraction from combined content+thinking ({len(combined):,} chars)")
                            try:
                                parsed = _parse_json_response(combined)
                            except (json.JSONDecodeError, ValueError):
                                pass

                    if parsed is None:
                        # Re-raise the original error to trigger retry
                        raise parse_error or json.JSONDecodeError(
                            "No valid JSON found in response",
                            content[:200] if content else "(empty)", 0
                        )
                    
                    # Attach thinking to result
                    if thinking_text and isinstance(parsed, dict):
                        parsed["_thinking"] = thinking_text
                    
                    # Attach tool usage metadata
                    if isinstance(parsed, dict):
                        parsed["_tool_call_count"] = tool_call_count
                        parsed["_tools_used"] = list(tools_used)
                    
                    await cb("llm_done", f"✓ {label} — Complete", {
                        "type": "llm_done",
                        "task": label,
                        "total_seconds": round(elapsed, 2),
                        "thinking_chars": len(thinking_text) if thinking_text else 0,
                    })
                    return parsed
                
                # Free-text response — attach thinking as prefix if present
                if thinking_text:
                    content = f"<!-- THINKING -->\n{thinking_text}\n<!-- /THINKING -->\n\n{content}"
                
                await cb("llm_done", f"✓ {label} — Complete ({len(content)} chars)", {
                    "type": "llm_done",
                    "task": label,
                    "total_seconds": round(elapsed, 2),
                    "response_length": len(content),
                    "thinking_chars": len(thinking_text) if thinking_text else 0,
                })
                return content

        except (json.JSONDecodeError, KeyError) as e:
            last_error = e
            elapsed = time.time() - t0
            await cb("llm_parse_error", f"{label} — JSON parse failed ({elapsed:.1f}s), will retry", {
                "type": "llm_parse_error",
                "task": label,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
                "attempt": attempt + 1,
            })

            # Log what the model actually returned for debugging
            logger.warning(f"[{label}] Failed content (first 2000 chars): {repr(content[:2000]) if content else '(empty)'}")
            if thinking_text:
                logger.warning(f"[{label}] Thinking was present ({len(thinking_text):,} chars) but JSON extraction failed")

            # Fallback: if structured output (schema) failed, drop to basic "json" mode
            if schema_enforced:
                logger.warning(f"[{label}] Schema-enforced output failed, falling back to basic JSON mode")
                format_param = "json"
                schema_enforced = False

            # Reset to clean messages for retry — do NOT append broken
            # tool call history or empty assistant content, as that poisons
            # subsequent attempts and guarantees further failures.
            if attempt < LLM_MAX_RETRIES - 1:
                messages = [m.copy() for m in original_messages]
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Please return ONLY a valid JSON object with the requested fields, no extra text or explanation."
                    ),
                })

                # On the last retry, disable tools entirely — the model
                # may be stuck in a tool-calling loop that never produces JSON
                if attempt == LLM_MAX_RETRIES - 2:
                    logger.warning(f"[{label}] Last retry — disabling tools to force direct JSON output")
                    use_tools = None
                    use_think = False
            continue
        except httpx.HTTPError as e:
            last_error = e
            elapsed = time.time() - t0
            await cb("llm_error", f"{label} — HTTP error: {e} ({elapsed:.1f}s)", {
                "type": "llm_error",
                "task": label,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
                "attempt": attempt + 1,
            })
            if attempt < LLM_MAX_RETRIES - 1:
                continue
            break

    await cb("llm_failed", f"{label} — Failed after {LLM_MAX_RETRIES} attempts", {
        "type": "llm_failed",
        "task": label,
        "error": str(last_error),
    })

    # Graceful degradation for JSON calls: return a minimal valid result
    # instead of crashing the entire pipeline
    if expect_json:
        logger.error(f"[{label}] Returning empty fallback JSON after {LLM_MAX_RETRIES} failed attempts")
        return {
            "checks": [],
            "group_score_deduction": 0,
            "_error": f"LLM failed after {LLM_MAX_RETRIES} attempts: {last_error}",
            "_fallback": True,
        }

    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts: {last_error}")


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from LLM response text.

    Attempts multiple extraction strategies in order:
      1. Direct parse
      2. Markdown code block extraction
      3. Greedy brace/bracket scanning
      4. Line-by-line JSON object detection (for mixed text+JSON)
    """
    import re
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response", "", 0)

    text = text.strip()

    # Strip any inline <think>...</think> blocks the model may emit
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also strip <|start_thinking|>...<|end_thinking|> variants
    text = re.sub(r'<\|start_thinking\|>.*?<\|end_thinking\|>', '', text, flags=re.DOTALL).strip()

    if not text:
        raise json.JSONDecodeError("Empty response after stripping think blocks", "", 0)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block — handle multiple code blocks
    code_block_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)```', re.DOTALL)
    for match in code_block_pattern.finditer(text):
        block = match.group(1).strip()
        if block and ('{' in block or '[' in block):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Try finding JSON object/array boundaries — scan all start positions
    # and try from each one to the farthest matching end bracket
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_positions = [i for i, c in enumerate(text) if c == start_char]
        for start in start_positions:
            end = len(text) - 1
            while end > start:
                if text[end] == end_char:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        pass
                end -= 1

    # Last resort: try to find JSON on individual lines (model may have
    # output text before/after a single-line JSON object)
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # ── JSON repair: fix common LLM output issues ──
    repaired = _attempt_json_repair(text)
    if repaired is not None:
        return repaired

    raise json.JSONDecodeError("No valid JSON found in response", text[:200], 0)


def _attempt_json_repair(text: str) -> dict | None:
    """Try to repair common JSON issues from LLM output.

    Fixes:
      - Trailing commas before } or ]
      - Unclosed brackets/braces (appends missing closers)
      - Trailing non-JSON text after the last }
      - Single-quoted strings → double-quoted
    Returns the parsed dict/list on success, None on failure.
    """
    import re

    # Find the first { and strip everything before it
    first_brace = text.find("{")
    if first_brace < 0:
        return None
    candidate = text[first_brace:]

    # Strip trailing non-JSON text after the last }
    last_brace = candidate.rfind("}")
    if last_brace >= 0:
        candidate = candidate[: last_brace + 1]

    # Fix trailing commas: ,} → } and ,] → ]
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    # Try parse after trailing-comma fix
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Count unmatched braces/brackets and append closers
    opens = 0
    open_sq = 0
    in_string = False
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_sq += 1
        elif ch == "]":
            open_sq -= 1

    if opens > 0 or open_sq > 0:
        candidate += "]" * max(open_sq, 0) + "}" * max(opens, 0)
        # Fix trailing commas again after appending closers
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════
# VISION LLM — for table-heavy document extraction using page images
# ═══════════════════════════════════════════════════════════════════

async def call_vision_llm(
    prompt: str,
    images: list[str],
    system_prompt: str = "",
    temperature: float = 0.1,
    expect_json: bool | dict = True,
    task_label: str = "",
    on_progress: LLMProgressCallback | None = None,
) -> dict | str:
    """Call vision model (qwen3-vl) with page images for extraction.

    Simpler than call_llm — no tool calling, no thinking, no multi-round.
    Sends base64-encoded page images to the vision model via Ollama's
    images API to extract structured data from table-heavy documents.

    Args:
        prompt: User prompt describing what to extract
        images: List of base64-encoded PNG images (one per page)
        system_prompt: System prompt for context
        temperature: LLM temperature
        expect_json: True for basic JSON mode, or a JSON Schema dict for structured outputs
        task_label: Human-readable label for this call
        on_progress: Async callback for progress updates

    Returns:
        Parsed JSON dict or raw string
    """
    cb = on_progress or _noop_cb
    label = task_label or "Vision LLM"

    # Build messages — images go in the user message
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Ollama vision API: images are base64 strings in the user message
    user_msg = {"role": "user", "content": prompt, "images": images}
    messages.append(user_msg)

    # Determine format parameter
    # NOTE: Always use basic JSON mode for vision calls — qwen3-vl:8b
    # handles simple schemas (classification) but returns empty responses
    # for complex nested schemas (extraction). The prompt guides the
    # output structure; _parse_json_response() handles JSON extraction.
    if expect_json:
        format_param = "json"
        schema_enforced = False
    else:
        format_param = ""
        schema_enforced = False

    prompt_chars = len(prompt) + (len(system_prompt) if system_prompt else 0)
    await cb("llm_start", f"{label}", {
        "type": "llm_start",
        "task": label,
        "model": "HATAD AI \u2726 Vision",
        "prompt_chars": prompt_chars,
        "image_count": len(images),
        "temperature": temperature,
        "expect_json": bool(expect_json),
        "schema_enforced": schema_enforced,
        "thinking_enabled": False,
        "tools_enabled": False,
    })

    last_error = None
    content = ""
    for attempt in range(LLM_MAX_RETRIES):
        t0 = time.time()

        if attempt > 0:
            backoff = min(2 ** attempt + random.uniform(0, 1), 30)
            await asyncio.sleep(backoff)
            await cb("llm_retry", f"{label} — Retry {attempt}/{LLM_MAX_RETRIES}", {
                "type": "llm_retry",
                "task": label,
                "attempt": attempt + 1,
                "reason": str(last_error),
            })

        await cb("llm_waiting", f"{label} — Sending {len(images)} page image(s) to HATAD AI Vision...", {
            "type": "llm_waiting",
            "task": label,
            "model": "HATAD AI ✦ Vision",
            "attempt": attempt + 1,
        })

        try:
            body = {
                "model": VISION_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 16384,  # Extraction output is compact JSON
                    "num_ctx": VISION_CONTEXT_WINDOW,
                },
                "format": format_param,
                "think": False,  # 8B vision model: thinking + format:json conflicts
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(VISION_TIMEOUT, connect=30.0)) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=body,
                )
                response.raise_for_status()
                elapsed = time.time() - t0
                result = response.json()
                message = result["message"]
                content = message.get("content", "")

                # qwen3-vl may put thinking in a separate field
                thinking_text = message.get("thinking", "")
                if thinking_text:
                    logger.debug(f"[{label}] Vision thinking: {len(thinking_text)} chars")

                # Debug empty responses
                if not content.strip():
                    logger.warning(
                        f"[{label}] Empty content from vision model. "
                        f"message_keys={list(message.keys())}, "
                        f"done={result.get('done')}, "
                        f"eval_count={result.get('eval_count', 0)}"
                    )

                # Parse metrics
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 0)
                tok_per_sec = (eval_count / (eval_duration / 1e9)) if eval_duration else 0

                await cb("llm_complete", f"{label} — Done ({elapsed:.1f}s, {tok_per_sec:.1f} tok/s)", {
                    "type": "llm_complete",
                    "task": label,
                    "model": "HATAD AI ✦ Vision",
                    "elapsed": elapsed,
                    "output_chars": len(content),
                    "eval_count": eval_count,
                    "tok_per_sec": round(tok_per_sec, 1),
                })

                # Parse JSON if expected
                if expect_json and content.strip():
                    try:
                        parsed = json.loads(content)
                        if thinking_text and isinstance(parsed, dict):
                            parsed["_thinking"] = thinking_text
                        return parsed
                    except json.JSONDecodeError as e:
                        try:
                            parsed = _parse_json_response(content)
                            if thinking_text and isinstance(parsed, dict):
                                parsed["_thinking"] = thinking_text
                            return parsed
                        except (json.JSONDecodeError, Exception):
                            last_error = f"JSON parse error: {e}"
                            logger.warning(f"[{label}] {last_error}, attempt {attempt + 1}")
                            continue
                elif expect_json and thinking_text:
                    # Content empty but thinking has text — try extracting JSON from thinking
                    logger.info(f"[{label}] Content empty, trying JSON from thinking ({len(thinking_text)} chars)")
                    try:
                        parsed = _parse_json_response(thinking_text)
                        if isinstance(parsed, dict):
                            parsed["_thinking"] = thinking_text
                        return parsed
                    except (json.JSONDecodeError, Exception):
                        last_error = "Empty content, thinking had no valid JSON"
                        logger.warning(f"[{label}] {last_error}, attempt {attempt + 1}")
                        continue
                elif not expect_json:
                    return content
                else:
                    last_error = "Empty response from vision model"
                    logger.warning(f"[{label}] {last_error}, attempt {attempt + 1}")
                    continue

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}"
            logger.warning(f"[{label}] {last_error}, attempt {attempt + 1}")
        except httpx.TimeoutException:
            last_error = f"Timeout ({VISION_TIMEOUT}s)"
            logger.warning(f"[{label}] {last_error}, attempt {attempt + 1}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[{label}] Error: {last_error}, attempt {attempt + 1}")

    # All retries exhausted
    raise RuntimeError(f"[{label}] Vision LLM failed after {LLM_MAX_RETRIES} attempts: {last_error}")


async def check_ollama_status() -> dict:
    """Check if Ollama is running and model is available.
    
    Results are cached for 120 seconds to avoid redundant HTTP calls
    during a single pipeline run.
    """
    global _ollama_status_cache, _ollama_status_ts
    now = time.time()
    if _ollama_status_cache is not None and (now - _ollama_status_ts) < 120:
        return _ollama_status_cache

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            result = {
                "status": "online",
                "models": model_names,
                "configured_model": "HATAD AI \u2726 Reasoning",
                "model_available": any(OLLAMA_MODEL in name for name in model_names),
                "vision_model": "HATAD AI \u2726 Vision",
                "vision_model_available": any(VISION_MODEL in name for name in model_names),
            }
            _ollama_status_cache = result
            _ollama_status_ts = now
            return result
    except Exception as e:
        return {"status": "offline", "error": str(e)}


# Cache for check_ollama_status
_ollama_status_cache: dict | None = None
_ollama_status_ts: float = 0.0


async def get_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Get embeddings for a batch of texts via Ollama /api/embed.

    Args:
        texts: List of text strings to embed
        model: Embedding model name (defaults to config.EMBED_MODEL)

    Returns:
        List of embedding vectors (list of floats)
    """
    from app.config import EMBED_MODEL, EMBED_BATCH_SIZE
    model = model or EMBED_MODEL

    all_embeddings: list[list[float]] = []

    # Process in batches to avoid overloading
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(batch):
                    logger.warning(
                        f"Embedding count mismatch: sent {len(batch)}, got {len(embeddings)}"
                    )
                all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Embedding batch {i // EMBED_BATCH_SIZE} failed: {e}")
            # Return zero vectors as fallback so indexing doesn't crash
            all_embeddings.extend([[0.0] * 768] * len(batch))

    return all_embeddings
