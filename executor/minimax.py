from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# MiniMax may report exhausted Token Plan availability as either 2062
# (inactive/no available plan) or 2067 (plan usage limit reached).
QUOTA_OR_BALANCE_CODES = {1008, 2056, 2062, 2067}


class MiniMaxClient:
    def __init__(self, config: dict[str, Any], root: Path, usage_log: Path):
        self.config = config
        self.root = root
        self.usage_log = usage_log
        self.last_request = 0.0
        self.active_key_index = 0

    def complete(self, messages: list[dict[str, str]], purpose: str, paper_id: str,
                 request_options: dict[str, Any] | None = None) -> dict[str, Any]:
        keys = [x.strip() for x in (self.root / self.config["key_file"]).read_text().splitlines() if x.strip()]
        if not keys:
            raise RuntimeError("No MiniMax credentials found")
        configured_indices = self.config.get("credential_indices")
        if configured_indices is None:
            allowed_indices = list(range(len(keys)))
        else:
            allowed_indices = sorted({int(index) - 1 for index in configured_indices})
            if not allowed_indices or any(index < 0 or index >= len(keys) for index in allowed_indices):
                raise RuntimeError("Configured MiniMax credential_indices are invalid")
        keys = [key for index, key in enumerate(keys) if index in allowed_indices]
        original_indices = [index for index in allowed_indices]
        if self.active_key_index not in range(len(keys)):
            self.active_key_index = 0
        request_body = {"model": self.config["model"], "messages": messages}
        if request_options:
            request_body.update(request_options)
        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        exhausted_keys: set[int] = set()
        key_order = [self.active_key_index] + [i for i in range(len(keys)) if i != self.active_key_index]
        for attempt in range(self.config["max_retries"]):
            for key_index in key_order:
                if key_index in exhausted_keys:
                    continue
                delay = max(0.0, self.config["min_seconds_between_requests"] - (time.monotonic() - self.last_request))
                if delay:
                    time.sleep(delay)
                req = urllib.request.Request(
                    self.config["api_endpoint"], data=body,
                    headers={"Authorization": "Bearer " + keys[key_index], "Content-Type": "application/json"},
                    method="POST",
                )
                started = time.monotonic()
                self.last_request = started
                response_meta: dict[str, Any] = {}
                try:
                    with urllib.request.urlopen(req, timeout=240) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("Malformed MiniMax response envelope")
                    base_resp = payload.get("base_resp")
                    if isinstance(base_resp, dict):
                        response_meta["base_status_code"] = base_resp.get("status_code")
                        response_meta["base_status_msg"] = str(base_resp.get("status_msg") or "")[:300]
                    choices = payload.get("choices")
                    response_meta["choices_type"] = type(choices).__name__
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        raise ValueError("Malformed MiniMax choices envelope")
                    choice = choices[0]
                    message = choice.get("message")
                    response_meta["message_type"] = type(message).__name__
                    if not isinstance(message, dict):
                        raise ValueError("Malformed MiniMax message envelope")
                    content = message.get("content") or ""
                    if not isinstance(content, str):
                        raise ValueError("Malformed MiniMax message content")
                    usage = payload.get("usage") or {}
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(), "paper_id": paper_id,
                        "purpose": purpose, "model": payload.get("model", self.config["model"]),
                        "credential_index": original_indices[key_index] + 1,
                        "total_tokens": usage.get("total_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "finish_reason": choice.get("finish_reason"),
                        "elapsed_seconds": round(time.monotonic() - started, 3), "attempt": attempt + 1,
                    }
                    with self.usage_log.open("a", encoding="utf-8") as log:
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    self.active_key_index = key_index
                    return {"content": content, "usage": usage, "model": event["model"],
                            "finish_reason": event["finish_reason"]}
                except urllib.error.HTTPError as e:
                    last_error = RuntimeError(f"MiniMax HTTP {e.code}")
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(), "paper_id": paper_id,
                        "purpose": purpose, "model": self.config["model"], "attempt": attempt + 1,
                        "credential_index": original_indices[key_index] + 1, "http_status": e.code, "status": "error",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                    with self.usage_log.open("a", encoding="utf-8") as log:
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    if e.code in QUOTA_OR_BALANCE_CODES:
                        exhausted_keys.add(key_index)
                        if len(exhausted_keys) == len(keys):
                            raise RuntimeError("MiniMax quota or balance exhausted for all configured credentials") from None
                    elif e.code not in (408, 425, 429, 500, 502, 503, 504, 529):
                        raise last_error from None
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                        KeyError, IndexError, TypeError, ValueError) as e:
                    last_error = RuntimeError(type(e).__name__)
                    event = {
                        "timestamp": datetime.now(timezone.utc).isoformat(), "paper_id": paper_id,
                        "purpose": purpose, "model": self.config["model"], "attempt": attempt + 1,
                        "credential_index": original_indices[key_index] + 1, "error_type": type(e).__name__, "status": "error",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "response_metadata": response_meta,
                    }
                    with self.usage_log.open("a", encoding="utf-8") as log:
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    try:
                        base_status_code = int(response_meta.get("base_status_code"))
                    except (TypeError, ValueError):
                        base_status_code = None
                    if base_status_code in QUOTA_OR_BALANCE_CODES:
                        exhausted_keys.add(key_index)
                        if len(exhausted_keys) == len(keys):
                            raise RuntimeError("MiniMax quota or balance exhausted for all configured credentials") from None
            if attempt + 1 < self.config["max_retries"]:
                time.sleep(min(90, 2 ** (attempt + 1)))
        raise last_error or RuntimeError("MiniMax request failed")
