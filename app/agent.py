"""
Scheduling agent using Groq SDK directly — no LangChain, no bind_tools.
Detects intent from user message, calls tools directly in Python.
"""

import json
import re
from datetime import datetime

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, update_schedule

client = Groq(api_key=GROQ_API_KEY)

TOOLS = {
    "get_schedule": get_schedule,
    "update_schedule": update_schedule,
}


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a scheduling assistant. Today is {today}.

Respond with ONLY a JSON object (no other text) to call a tool:

To retrieve schedule:
{{"tool": "get_schedule", "args": {{"query": "YYYY-MM-DD"}}}}

To add/update/delete an event:
{{"tool": "update_schedule", "args": {{"action": "add", "title": "Event title", "date": "YYYY-MM-DD", "time": "HH:MM", "event_type": "meeting", "description": ""}}}}

Rules:
1. Resolve relative dates to exact YYYY-MM-DD using today ({today}).
2. For schedule queries output ONLY the get_schedule JSON.
3. For adding/updating/deleting output ONLY the update_schedule JSON.
4. After receiving tool results answer the user naturally.
5. Never invent schedule data.
6. If no tool is needed answer directly."""


def _extract_tool_call(text: str) -> dict | None:
    text = text.strip()
    try:
        d = json.loads(text)
        if "tool" in d:
            return d
    except Exception:
        pass
    match = re.search(r'\{[^{}]*"tool"\s*:[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def run_agent(user_input: str) -> str:
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_input},
    ]

    for _ in range(5):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        tool_call = _extract_tool_call(content)
        if not tool_call:
            return content.strip() or "I couldn't process that request."

        name = tool_call.get("tool", "")
        args = tool_call.get("args", {})
        tool_fn = TOOLS.get(name)

        if tool_fn:
            try:
                result = tool_fn.invoke(args)
            except Exception as e:
                result = f"Tool error: {e}"
        else:
            result = f"Unknown tool: {name}"

        messages.append({"role": "user", "content": f"Tool result for {name}:\n{result}"})

    return "I reached the maximum steps. Please try a more specific question."
