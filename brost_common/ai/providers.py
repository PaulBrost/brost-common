"""
LLM provider implementations for the Brost common library.

All providers expose a uniform chat(messages, **opts) -> ChatResponse interface.
Provider-specific quirks (Anthropic system-prompt placement and current-gen
sampling-parameter omission, Azure deployment URLs, GPT-5/o1 parameter
naming) are isolated here so callers stay provider-agnostic.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from .response import ChatResponse, ToolCall

logger = logging.getLogger(__name__)


def _uses_completion_tokens(model: str) -> bool:
    """True for models that require max_completion_tokens instead of max_tokens."""
    m = model.lower()
    return m.startswith('o1') or '-o1' in m or 'gpt-5' in m


def _anthropic_omits_sampling(model: str) -> bool:
    """Opus 4.7+, Sonnet 5, and Fable 5 / Mythos 5 reject temperature/top_p/top_k
    (HTTP 400). Older Claude models still accept them."""
    m = model.lower()
    return any(tok in m for tok in ('opus-4-7', 'opus-4-8', 'sonnet-5', 'fable-5', 'mythos-5'))


class BaseProvider:
    """Abstract base. Subclasses must implement chat()."""

    default_base_url: str = ''

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get('model', '')
        self.api_key = config.get('api_key', '')
        self.base_url = (config.get('base_url') or self.default_base_url).rstrip('/') + '/'

    def chat(self, messages: list[dict], *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResponse:
        raise NotImplementedError

    def health_check(self) -> bool:
        try:
            self.chat([{'role': 'user', 'content': 'ping'}], max_tokens=8)
            return True
        except Exception:
            return False


class AnthropicProvider(BaseProvider):
    default_base_url = 'https://api.anthropic.com/v1/'

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResponse:
        # Anthropic: system prompt goes top-level, not in the messages array.
        # Tool-call schema is also different from OpenAI format.
        system_text = ''
        api_messages = []
        for m in messages:
            role = m.get('role', '')
            if role == 'system':
                system_text += (m.get('content') or '') + '\n'
            elif role == 'tool':
                api_messages.append({
                    'role': 'user',
                    'content': [{
                        'type': 'tool_result',
                        'tool_use_id': m.get('tool_call_id', ''),
                        'content': m.get('content', ''),
                    }],
                })
            elif role == 'assistant' and m.get('tool_calls'):
                blocks: list[dict[str, Any]] = []
                if m.get('content'):
                    blocks.append({'type': 'text', 'text': m['content']})
                for tc in m['tool_calls']:
                    blocks.append({
                        'type': 'tool_use',
                        'id': tc['id'],
                        'name': tc['function']['name'],
                        'input': json.loads(tc['function']['arguments']),
                    })
                api_messages.append({'role': 'assistant', 'content': blocks})
            else:
                api_messages.append({'role': role, 'content': m.get('content', '')})

        body: dict[str, Any] = {
            'model': self.model or 'claude-sonnet-4-6',
            'max_tokens': max_tokens,
            'messages': api_messages,
        }
        if not _anthropic_omits_sampling(body['model']):
            body['temperature'] = temperature
        if system_text.strip():
            body['system'] = system_text.strip()
        if tools:
            body['tools'] = [
                {
                    'name': (fn := t.get('function', t))['name'],
                    'description': fn.get('description', ''),
                    'input_schema': fn.get('parameters', {'type': 'object', 'properties': {}}),
                }
                for t in tools
            ]

        url = self.base_url + 'messages'
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
        }

        start = time.time()
        data = self._post_with_retry(url, headers, body)
        duration_ms = int((time.time() - start) * 1000)

        stop_reason = data.get('stop_reason', 'end_turn')
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get('content', []):
            if block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif block.get('type') == 'tool_use':
                tool_calls.append(ToolCall(
                    id=block['id'],
                    name=block['name'],
                    arguments=block.get('input', {}),
                ))

        return ChatResponse(
            content=''.join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason='tool_calls' if stop_reason == 'tool_use' else 'stop',
            raw=data,
            model=data.get('model', body['model']),
            duration_ms=duration_ms,
        )

    def _post_with_retry(self, url, headers, body) -> dict:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=120)
                if resp.status_code == 429:
                    delay = int(resp.headers.get('retry-after', '2'))
                    time.sleep(min(delay, 10) * (attempt + 1))
                    last_err = RuntimeError('rate-limited')
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_err = exc
                if any(s in str(exc) for s in ('401', '403')):
                    raise
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    continue
                raise
        raise last_err  # type: ignore[misc]


class OpenAICompatProvider(BaseProvider):
    """OpenAI and OpenAI-compatible endpoints (Ollama, OpenRouter, Azure AI Foundry, etc.)."""

    default_base_url = 'https://api.openai.com/v1/'

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResponse:
        model = self.model or 'gpt-4o-mini'
        new_style = _uses_completion_tokens(model)

        body: dict[str, Any] = {'model': model, 'messages': messages}
        if not new_style:
            body['temperature'] = temperature
        body['max_completion_tokens' if new_style else 'max_tokens'] = max_tokens
        if tools:
            body['tools'] = tools

        url = self.base_url + 'chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        start = time.time()
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=120)
                if resp.status_code == 429:
                    delay = int(resp.headers.get('retry-after', '2'))
                    time.sleep(min(delay, 10) * (attempt + 1))
                    last_err = RuntimeError('rate-limited')
                    continue
                resp.raise_for_status()
                data = resp.json()
                choice = data['choices'][0]
                msg = choice.get('message', {})
                tool_calls: list[ToolCall] = []
                for tc in msg.get('tool_calls') or []:
                    fn = tc.get('function', {})
                    args_raw = fn.get('arguments', '{}')
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {'_raw': args_raw}
                    tool_calls.append(ToolCall(
                        id=tc.get('id', ''),
                        name=fn.get('name', ''),
                        arguments=args,
                    ))
                return ChatResponse(
                    content=msg.get('content') or None,
                    tool_calls=tool_calls,
                    finish_reason=choice.get('finish_reason', 'stop'),
                    raw=data,
                    model=data.get('model', model),
                    duration_ms=int((time.time() - start) * 1000),
                )
            except Exception as exc:
                last_err = exc
                if any(s in str(exc) for s in ('401', '403')):
                    raise
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    continue
                raise
        raise last_err  # type: ignore[misc]


class AzureProvider(BaseProvider):
    """Azure OpenAI — deployment-based URL, api-key header auth.

    Supports both the legacy Chat Completions API and the newer Responses API
    (required for GPT-5.5 and later models on Azure AI Foundry).

    Responses API is used when:
      - azure_base_url already contains '/openai/v1'  (user sets the Foundry endpoint directly), OR
      - the deployment name contains 'gpt-5.5'

    Chat Completions API is used for all other deployments.
    """

    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', '')
        # Accept azure_base_url (site_ai / gb_grooves) or base_url (DjangoTemplate)
        self.azure_base = (
            config.get('azure_base_url') or config.get('base_url') or ''
        ).rstrip('/')
        self.deployment = config.get('azure_deployment', '')
        self.api_version = config.get('azure_api_version') or '2024-02-15-preview'

    def _use_responses_api(self) -> bool:
        return (
            '/openai/v1' in self.azure_base
            or 'gpt-5.5' in self.deployment.lower()
            or 'gpt-5.5' in self.model.lower()
        )

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResponse:
        if not self.azure_base:
            raise ValueError("AzureProvider requires azure_base_url in config")
        if not self.deployment:
            raise ValueError("AzureProvider requires azure_deployment in config")

        if self._use_responses_api():
            return self._chat_responses(messages, tools=tools, max_tokens=max_tokens)
        return self._chat_completions(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)

    def _chat_completions(self, messages, *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResponse:
        """Legacy Chat Completions API — all models except GPT-5.5+."""
        url = (
            f"{self.azure_base}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

        d_lower = self.deployment.lower()
        m_lower = self.model.lower()
        is_o1 = 'o1' in d_lower or 'o1' in m_lower
        new_style = is_o1 or 'gpt-5' in d_lower or 'gpt-5' in m_lower

        body: dict[str, Any] = {'messages': messages}
        if not new_style:
            body['temperature'] = temperature
        body['max_completion_tokens' if new_style else 'max_tokens'] = max_tokens
        if tools:
            body['tools'] = tools

        headers = {'Content-Type': 'application/json', 'api-key': self.api_key}

        start = time.time()
        resp = requests.post(url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        choice = data['choices'][0]
        msg = choice.get('message', {})
        tool_calls: list[ToolCall] = []
        for tc in msg.get('tool_calls') or []:
            fn = tc.get('function', {})
            args_raw = fn.get('arguments', '{}')
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {'_raw': args_raw}
            tool_calls.append(ToolCall(
                id=tc.get('id', ''),
                name=fn.get('name', ''),
                arguments=args,
            ))

        return ChatResponse(
            content=msg.get('content') or None,
            tool_calls=tool_calls,
            finish_reason=choice.get('finish_reason', 'stop'),
            raw=data,
            model=data.get('model', self.deployment),
            duration_ms=int((time.time() - start) * 1000),
        )

    def _chat_responses(self, messages, *, tools=None, max_tokens=4096) -> ChatResponse:
        """Responses API — required for GPT-5.5+ on Azure AI Foundry.

        URL shape:  {base}/openai/v1/responses
        If azure_base already ends with /openai/v1 we append /responses directly;
        otherwise we insert /openai/v1/responses after the host.
        """
        if '/openai/v1' in self.azure_base:
            # User already included the /openai/v1 path segment
            base = self.azure_base.rstrip('/')
            url = f"{base}/responses"
        else:
            url = f"{self.azure_base}/openai/v1/responses"

        # Responses API: system messages go in the 'instructions' param; everything
        # else goes in 'input' as-is (role/content dicts are accepted directly).
        system_parts = [m['content'] for m in messages if m.get('role') == 'system']
        input_messages = [m for m in messages if m.get('role') != 'system']

        body: dict[str, Any] = {
            'model': self.deployment,
            'input': input_messages,
            'max_output_tokens': max_tokens,
        }
        if system_parts:
            body['instructions'] = '\n'.join(str(p) for p in system_parts)
        if tools:
            body['tools'] = tools

        headers = {'Content-Type': 'application/json', 'api-key': self.api_key}

        start = time.time()
        resp = requests.post(url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        duration_ms = int((time.time() - start) * 1000)

        # Parse output items — text and tool calls have different item types.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in data.get('output', []):
            item_type = item.get('type')
            if item_type == 'message':
                for block in item.get('content', []):
                    if block.get('type') == 'output_text':
                        text_parts.append(block.get('text', ''))
            elif item_type == 'function_call':
                args_raw = item.get('arguments', '{}')
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {'_raw': args_raw}
                tool_calls.append(ToolCall(
                    id=item.get('call_id', item.get('id', '')),
                    name=item.get('name', ''),
                    arguments=args,
                ))

        finish = 'tool_calls' if tool_calls and not text_parts else 'stop'
        return ChatResponse(
            content=''.join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            raw=data,
            model=data.get('model', self.deployment),
            duration_ms=duration_ms,
        )


class MockProvider(BaseProvider):
    default_base_url = 'http://mock.invalid/'

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=4096) -> ChatResponse:
        last_user = next(
            (m['content'] for m in reversed(messages) if m.get('role') == 'user'), ''
        )
        return ChatResponse(content=f'[mock] echo: {str(last_user)[:200]}', model='mock')

    def health_check(self) -> bool:
        return True
