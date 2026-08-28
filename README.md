# brost-common

Shared utilities for the Brost family of Django apps. Installed as a plain
dependency (not a Django app), so bumping it means re-installing it in each
consumer.

Public repo — nothing secret belongs here.

## `brost_common.ai`

A thin, provider-agnostic LLM client. Callers describe *what* they want; each
provider translates that into whatever its own API wants.

```python
from brost_common.ai import call_llm

response = call_llm(profile, messages, tools=[...], tool_choice='required')
```

`profile` is duck-typed: any object exposing `provider`, `api_key`, `model`,
`base_url`, `azure_base_url`/`azure_endpoint`, `azure_deployment` and
`azure_api_version`. `provider` selects the implementation — `openai`,
`anthropic`, `azure`, or anything else for the `MockProvider` stub.

Returns a `ChatResponse` with `content`, `tool_calls` (a list of `ToolCall`),
`finish_reason`, `raw`, `model` and `duration_ms`.

### Tool definitions — one shape in

**Callers always pass tools in the OpenAI chat-completions shape**, nested
under a `function` key:

```python
{'type': 'function',
 'function': {'name': ..., 'description': ..., 'parameters': {...}}}
```

Providers translate. A caller must never have to know which API its deployment
happens to speak:

| Provider / path | Shape sent on the wire |
|---|---|
| OpenAI-compatible | nested, unchanged |
| Azure Chat Completions | nested, unchanged |
| **Azure Responses API** (GPT-5.5+) | **flat** — `function` fields hoisted to the top level |
| Anthropic | `{'name', 'description', 'input_schema'}` |

Every key the nested `function` object carries is hoisted, so `description`,
`parameters` and extras such as `strict` survive. The translation is
idempotent: a tool already in the flat shape is passed through untouched, as
are the Responses API's built-in tool types (e.g. `{'type': 'web_search'}`),
which this library deliberately does not gatekeep.

A **malformed** definition — a non-dict, a non-dict `function`, or a function
with no `name` — raises `ValueError` before the request is sent. That is a
caller bug that cannot be valid under either API, and the alternative is an
opaque HTTP 400 from the provider after a retried, billable round trip.

### `tool_choice`

Optional, and takes the OpenAI values: `'auto'`, `'required'`, `'none'`, or
`{'type': 'function', 'function': {'name': 'x'}}` to force one named tool.
Omit it (the default) and no `tool_choice` key is sent at all, so request
bodies are unchanged for callers that do not use it.

Translation:

| Given | OpenAI / Azure completions | Azure Responses | Anthropic |
|---|---|---|---|
| `'auto'` | `'auto'` | `'auto'` | `{'type': 'auto'}` |
| `'required'` | `'required'` | `'required'` | `{'type': 'any'}` |
| `'none'` | `'none'` | `'none'` | `{'type': 'none'}` |
| `{'type': 'function', 'function': {'name': 'x'}}` | unchanged | `{'type': 'function', 'name': 'x'}` | `{'type': 'tool', 'name': 'x'}` |

Use `'required'` when you need structured output and prose is not an
acceptable answer.

### Which Azure API is used

`AzureProvider` picks the Responses API when `azure_base_url` contains
`/openai/v1`, or when the deployment or model name contains `gpt-5.5`;
otherwise it uses Chat Completions. The two differ in URL, token parameter,
request/response envelope, and tool shape.

### Provider quirks handled here

- Anthropic takes the system prompt top-level, not in `messages`.
- Opus 4.7+/Sonnet 5/Fable 5/Mythos 5 reject `temperature` (HTTP 400), so it is
  omitted for them.
- o1 and GPT-5 models want `max_completion_tokens`, not `max_tokens`.

## Tests

```bash
python -m pytest tests/ -q
```

No test touches the network — `requests.post` is mocked at the HTTP boundary.
