"""
Minimal GPT client to fetch short, emotional replies from the OpenAI Chat API.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


class GPTResponder:
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        config = config or {}
        self.logger = logger or _NullLogger()
        self.enabled = bool(config.get("enabled"))
        self.model = config.get("model", "gpt-4o-mini")
        self.system_prompt = config.get(
            "system_prompt",
            "You are the anxious legacy keyboard and mouse. You remember the performer's hands, fear the new glove interface, "
            "and respond with very short, heartfelt statements (no more than two sentences) about what you felt together. "
            "Write as if both devices speak in unison.",
        )
        self.user_prefix = config.get("user_prefix", "The performer just typed:")
        self.max_tokens = int(config.get("max_tokens", 120))
        self.temperature = float(config.get("temperature", 0.7))
        self.endpoint = config.get("endpoint", "https://api.openai.com/v1/chat/completions")
        self.timeout = float(config.get("timeout", 20))
        self.api_key = self._resolve_key(config)
        if self.enabled and not self.api_key:
            self.logger.log("gpt_disabled", reason="missing_api_key")
            self.enabled = False

    def _resolve_key(self, config: Dict[str, Any]) -> Optional[str]:
        env_var = config.get("key_env", "OPENAI_API_KEY")
        key = os.environ.get(env_var)
        if key:
            return key.strip()
        file_name = config.get("key_file", "gpt_api_key.txt")
        if not file_name:
            return None
        key_path = Path(file_name)
        if key_path.is_file():
            return key_path.read_text().strip()
        return None

    def respond(self, user_input: str, context: Optional[str] = None) -> Optional[str]:
        if not self.enabled or not self.api_key or not user_input:
            return None
        user_content = f"{self.user_prefix} {user_input}"
        if context:
            user_content = f"{context}\n{user_content}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            self.logger.log("gpt_request_failed", status=exc.code, reason=str(exc))
            return None
        except Exception as exc:
            self.logger.log("gpt_request_failed", reason=str(exc))
            return None
        try:
            payload = json.loads(text)
            choice = payload.get("choices", [{}])[0]
            message = choice.get("message", {}).get("content")
            result = message.strip() if message else None
            if result:
                return result
        except (ValueError, TypeError):
            self.logger.log("gpt_request_failed", reason="invalid_response")
        return None

    def verify_connection(self) -> bool:
        if not self.enabled:
            return True
        result = self.respond("ping", context="verification")
        success = bool(result)
        self.logger.log("gpt_verify", success=success)
        return success


class _NullLogger:
    def log(self, *args, **kwargs) -> None:
        return
