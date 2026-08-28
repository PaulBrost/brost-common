"""Tests for the provider tool-shape translation.

Every test mocks requests.post, so nothing here touches the network.
Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from brost_common.ai import AnthropicProvider, AzureProvider, OpenAICompatProvider, ToolCall
from brost_common.ai.providers import _responses_tool

# The canonical shape every caller of this library passes.
NESTED_TOOL = {
    'type': 'function',
    'function': {
        'name': 'record_score',
        'description': 'Record the score for a submission.',
        'parameters': {
            'type': 'object',
            'properties': {'score': {'type': 'integer'}},
            'required': ['score'],
        },
        'strict': True,
    },
}

# The shape the Responses API actually wants.
FLAT_TOOL = {
    'type': 'function',
    'name': 'record_score',
    'description': 'Record the score for a submission.',
    'parameters': {
        'type': 'object',
        'properties': {'score': {'type': 'integer'}},
        'required': ['score'],
    },
    'strict': True,
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def capture(payload):
    """Patch requests.post, returning (context manager, list-of-bodies)."""
    bodies = []

    def _post(url, json=None, headers=None, timeout=None):
        bodies.append(json)
        return FakeResponse(payload)

    return patch('brost_common.ai.providers.requests.post', side_effect=_post), bodies


RESPONSES_PAYLOAD = {
    'model': 'gpt-5.5',
    'output': [{
        'type': 'function_call',
        'call_id': 'call_abc123',
        'name': 'record_score',
        'arguments': '{"score": 7}',
    }],
}

COMPLETIONS_PAYLOAD = {
    'model': 'gpt-4o',
    'choices': [{
        'finish_reason': 'tool_calls',
        'message': {
            'content': None,
            'tool_calls': [{
                'id': 'call_abc123',
                'function': {'name': 'record_score', 'arguments': '{"score": 7}'},
            }],
        },
    }],
}

AZURE_RESPONSES_CFG = {
    'api_key': 'k',
    'azure_base_url': 'https://example.openai.azure.com',
    'azure_deployment': 'gpt-5.5-chat',
}
AZURE_COMPLETIONS_CFG = {
    'api_key': 'k',
    'azure_base_url': 'https://example.openai.azure.com',
    'azure_deployment': 'gpt-4o',
}


def azure_responses_body(**kwargs):
    ctx, bodies = capture(RESPONSES_PAYLOAD)
    with ctx:
        AzureProvider(AZURE_RESPONSES_CFG).chat([{'role': 'user', 'content': 'hi'}], **kwargs)
    return bodies[0]


def azure_completions_body(**kwargs):
    ctx, bodies = capture(COMPLETIONS_PAYLOAD)
    with ctx:
        AzureProvider(AZURE_COMPLETIONS_CFG).chat([{'role': 'user', 'content': 'hi'}], **kwargs)
    return bodies[0]


# --- routing sanity: the two configs really do take different paths ---------

def test_deployment_selects_the_expected_api():
    assert AzureProvider(AZURE_RESPONSES_CFG)._use_responses_api() is True
    assert AzureProvider(AZURE_COMPLETIONS_CFG)._use_responses_api() is False


# --- the bug: nested -> flat on the Responses path --------------------------

def test_responses_flattens_a_nested_tool():
    body = azure_responses_body(tools=[NESTED_TOOL])
    assert body['tools'] == [FLAT_TOOL]
    assert 'function' not in body['tools'][0]


def test_responses_preserves_description_parameters_and_strict():
    tool = azure_responses_body(tools=[NESTED_TOOL])['tools'][0]
    assert tool['description'] == NESTED_TOOL['function']['description']
    assert tool['parameters'] == NESTED_TOOL['function']['parameters']
    assert tool['strict'] is True


def test_responses_leaves_an_already_flat_tool_unchanged():
    assert azure_responses_body(tools=[FLAT_TOOL])['tools'] == [FLAT_TOOL]


def test_translation_is_idempotent():
    assert _responses_tool(_responses_tool(NESTED_TOOL)) == FLAT_TOOL


def test_responses_passes_builtin_tool_types_through():
    builtin = {'type': 'web_search'}
    assert azure_responses_body(tools=[builtin])['tools'] == [builtin]


def test_caller_tool_list_is_not_mutated():
    before = json.dumps(NESTED_TOOL, sort_keys=True)
    azure_responses_body(tools=[NESTED_TOOL])
    assert json.dumps(NESTED_TOOL, sort_keys=True) == before


# --- malformed input -------------------------------------------------------

@pytest.mark.parametrize('bad', [
    'record_score',                                   # not a dict
    {'type': 'function', 'function': 'record_score'},  # function is not a dict
    {'type': 'function', 'function': {'description': 'no name'}},  # no name
])
def test_malformed_tool_raises_before_any_request(bad):
    ctx, bodies = capture(RESPONSES_PAYLOAD)
    with ctx:
        with pytest.raises(ValueError):
            AzureProvider(AZURE_RESPONSES_CFG).chat([{'role': 'user', 'content': 'hi'}], tools=[bad])
    assert bodies == []  # failed before spending a request


# --- the chat-completions path must be untouched ---------------------------

def test_completions_still_sends_the_nested_shape():
    body = azure_completions_body(tools=[NESTED_TOOL])
    assert body['tools'] == [NESTED_TOOL]
    assert body['tools'][0]['function']['name'] == 'record_score'


def test_openai_compat_still_sends_the_nested_shape():
    ctx, bodies = capture(COMPLETIONS_PAYLOAD)
    with ctx:
        OpenAICompatProvider({'api_key': 'k', 'model': 'gpt-4o'}).chat(
            [{'role': 'user', 'content': 'hi'}], tools=[NESTED_TOOL]
        )
    assert bodies[0]['tools'] == [NESTED_TOOL]


# --- response parsing still yields ToolCall objects ------------------------

def test_responses_parsing_yields_tool_calls():
    ctx, _ = capture(RESPONSES_PAYLOAD)
    with ctx:
        result = AzureProvider(AZURE_RESPONSES_CFG).chat(
            [{'role': 'user', 'content': 'hi'}], tools=[NESTED_TOOL]
        )
    assert result.tool_calls == [ToolCall(id='call_abc123', name='record_score', arguments={'score': 7})]
    assert result.finish_reason == 'tool_calls'


def test_completions_parsing_yields_tool_calls():
    ctx, _ = capture(COMPLETIONS_PAYLOAD)
    with ctx:
        result = AzureProvider(AZURE_COMPLETIONS_CFG).chat(
            [{'role': 'user', 'content': 'hi'}], tools=[NESTED_TOOL]
        )
    assert result.tool_calls == [ToolCall(id='call_abc123', name='record_score', arguments={'score': 7})]


# --- tool_choice -----------------------------------------------------------

def test_tool_choice_is_absent_by_default():
    assert 'tool_choice' not in azure_responses_body(tools=[NESTED_TOOL])
    assert 'tool_choice' not in azure_completions_body(tools=[NESTED_TOOL])


@pytest.mark.parametrize('value', ['auto', 'required', 'none'])
def test_string_tool_choice_passes_through_unchanged_on_openai_shaped_apis(value):
    assert azure_completions_body(tools=[NESTED_TOOL], tool_choice=value)['tool_choice'] == value
    assert azure_responses_body(tools=[NESTED_TOOL], tool_choice=value)['tool_choice'] == value


def test_named_tool_choice_is_flattened_for_responses():
    choice = {'type': 'function', 'function': {'name': 'record_score'}}
    body = azure_responses_body(tools=[NESTED_TOOL], tool_choice=choice)
    assert body['tool_choice'] == {'type': 'function', 'name': 'record_score'}


def test_named_tool_choice_stays_nested_for_completions():
    choice = {'type': 'function', 'function': {'name': 'record_score'}}
    body = azure_completions_body(tools=[NESTED_TOOL], tool_choice=choice)
    assert body['tool_choice'] == choice


ANTHROPIC_PAYLOAD = {'model': 'claude-sonnet-5', 'stop_reason': 'end_turn', 'content': []}


def anthropic_body(**kwargs):
    ctx, bodies = capture(ANTHROPIC_PAYLOAD)
    with ctx:
        AnthropicProvider({'api_key': 'k', 'model': 'claude-sonnet-5'}).chat(
            [{'role': 'user', 'content': 'hi'}], **kwargs
        )
    return bodies[0]


@pytest.mark.parametrize('given,expected', [
    ('auto', {'type': 'auto'}),
    ('required', {'type': 'any'}),
    ('none', {'type': 'none'}),
    ({'type': 'function', 'function': {'name': 'record_score'}}, {'type': 'tool', 'name': 'record_score'}),
    ({'type': 'any'}, {'type': 'any'}),  # already Anthropic-shaped
])
def test_anthropic_tool_choice_translation(given, expected):
    assert anthropic_body(tools=[NESTED_TOOL], tool_choice=given)['tool_choice'] == expected


def test_anthropic_omits_tool_choice_by_default():
    assert 'tool_choice' not in anthropic_body(tools=[NESTED_TOOL])


def test_unsupported_tool_choice_string_raises():
    with pytest.raises(ValueError):
        anthropic_body(tools=[NESTED_TOOL], tool_choice='mandatory')


# --- dispatch layer --------------------------------------------------------

def test_call_llm_forwards_tool_choice():
    from types import SimpleNamespace

    from brost_common.ai import call_llm

    profile = SimpleNamespace(
        provider='azure', api_key='k', model='',
        azure_base_url='https://example.openai.azure.com',
        azure_deployment='gpt-5.5-chat', azure_api_version='',
    )
    ctx, bodies = capture(RESPONSES_PAYLOAD)
    with ctx:
        call_llm(profile, [{'role': 'user', 'content': 'hi'}],
                 tools=[NESTED_TOOL], tool_choice='required')
    assert bodies[0]['tools'] == [FLAT_TOOL]
    assert bodies[0]['tool_choice'] == 'required'
