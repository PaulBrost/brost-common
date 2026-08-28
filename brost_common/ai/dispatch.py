"""
Duck-typed dispatch helper.

call_llm(profile, messages, ...) accepts any object whose attributes provide
provider config — a Django model instance, a SimpleNamespace, or a dataclass.
Maps provider type → provider class and delegates.
"""
from __future__ import annotations

from .providers import AnthropicProvider, AzureProvider, MockProvider, OpenAICompatProvider
from .response import ChatResponse

_PROVIDER_MAP = {
    'openai': OpenAICompatProvider,
    'anthropic': AnthropicProvider,
    'azure': AzureProvider,
}


def call_llm(
    profile,
    messages: list[dict],
    tools=None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tool_choice=None,
) -> ChatResponse:
    """
    Call an LLM using a duck-typed profile object.

    profile must have a 'provider' attribute ('openai', 'anthropic', 'azure', or 'stub').
    Reads api_key, model, base_url, azure_base_url / azure_endpoint,
    azure_deployment, azure_api_version via getattr with empty-string defaults.

    tools are passed in the OpenAI chat-completions shape
    ({'type': 'function', 'function': {'name', 'description', 'parameters'}});
    each provider translates that to its own API's shape.

    tool_choice is optional and takes the OpenAI values -- 'auto', 'required',
    'none', or {'type': 'function', 'function': {'name': 'x'}} to force one
    named tool. Omitted by default, so the request body is unchanged for
    callers that do not use it.
    """
    provider_type = getattr(profile, 'provider', 'stub')
    config = {
        'api_key': getattr(profile, 'api_key', ''),
        'model': getattr(profile, 'model', ''),
        'base_url': getattr(profile, 'base_url', ''),
        # Support both azure_base_url (site_ai/gb_grooves) and azure_endpoint (tech_planner)
        'azure_base_url': (
            getattr(profile, 'azure_base_url', '')
            or getattr(profile, 'azure_endpoint', '')
        ),
        'azure_deployment': getattr(profile, 'azure_deployment', ''),
        'azure_api_version': getattr(profile, 'azure_api_version', ''),
    }
    cls = _PROVIDER_MAP.get(provider_type)
    if cls is None:
        return MockProvider(config).chat(messages)
    return cls(config).chat(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Alias kept for backward compatibility with site_ai callers
call_llm_with_profile = call_llm
