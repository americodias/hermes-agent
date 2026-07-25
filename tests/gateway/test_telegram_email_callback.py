"""Tests for the Telegram adapter's `em:` email-manager approval callbacks.

The `em:` payload contract is canonical in the email-manager repo
(docs/decisions/ws-t-telegram-approvals.md, "em: callback payload contract",
mirrored by email_manager.control.encode_email_callback/decode_email_callback):
closed grammar `em:approve:<ref>:<action>` / `em:dismiss:<ref>`, <=64 bytes,
fail-closed decode, the same authorization check as ea:/gt: callbacks, and
dispatch of the email-manager-operator plugin with argv identical to the typed
`/email tg-approve <ref> <action>` / `/email tg-dismiss <ref>` command — never
through the model loop.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _adapter_class():
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ModuleNotFoundError:  # PR branch before Telegram plugin extraction
        from gateway.platforms.telegram import TelegramAdapter
    return TelegramAdapter


def _make_adapter(callback_auth=None):
    TelegramAdapter = _adapter_class()
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    if callback_auth is not None:
        adapter._is_callback_user_authorized = callback_auth
    return adapter


def _make_query(data, *, user_id=111, chat_id=-100, chat_type="group"):
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=user_id, first_name="Test"),
        message=SimpleNamespace(
            chat_id=chat_id,
            chat=SimpleNamespace(id=chat_id, type=chat_type),
            message_thread_id=None,
            text="proposal",
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )


def _make_update(query):
    return SimpleNamespace(callback_query=query)


# --- decode: mirrors email_manager.control.decode_email_callback -------------


def test_decode_approve_yields_typed_command_argv():
    decode = _adapter_class()._decode_email_callback
    assert decode("em:approve:tg-123:archive") == ("tg-approve", "tg-123", "archive")


@pytest.mark.parametrize(
    "action", ["archive", "mark_read", "archive_document", "unsubscribe_oneclick"]
)
def test_decode_accepts_every_closed_action(action):
    decode = _adapter_class()._decode_email_callback
    assert decode(f"em:approve:tg-9:{action}") == ("tg-approve", "tg-9", action)


def test_decode_dismiss_yields_typed_command_argv():
    decode = _adapter_class()._decode_email_callback
    assert decode("em:dismiss:tg-123") == ("tg-dismiss", "tg-123")


@pytest.mark.parametrize(
    "payload",
    [
        None,
        123,
        "",
        "em",
        "em:",
        "em:approve",
        "em:approve:tg-123",  # approve missing action
        "em:approve:tg-123:archive:extra",  # approve wrong arity
        "em:dismiss:tg-123:archive",  # dismiss wrong arity
        "em:execute:tg-123",  # unknown verb
        "EM:approve:tg-123:archive",  # wrong prefix case
        "ea:once:42",  # other prefix
        "em:approve:tg-123:trash",  # action outside the closed set
        "em:approve:tg-123:ARCHIVE",  # action case outside the closed set
        "em:approve:.bad-ref:archive",  # ref outside identifier grammar
        "em:approve::archive",  # empty ref
        "em:dismiss:",  # empty ref
        "em:approve:tg-123:" + "a" * 64,  # oversized (>64 bytes)
    ],
)
def test_decode_fails_closed(payload):
    decode = _adapter_class()._decode_email_callback
    assert decode(payload) is None


def test_decode_64_byte_boundary():
    decode = _adapter_class()._decode_email_callback
    ref = "r" * (64 - len("em:dismiss:"))
    exactly_64 = f"em:dismiss:{ref}"
    assert len(exactly_64.encode("utf-8")) == 64
    assert decode(exactly_64) == ("tg-dismiss", ref)
    assert decode(exactly_64 + "r") is None


# --- routing + authorization + dispatch ---------------------------------------


@pytest.mark.asyncio
async def test_approve_press_dispatches_plugin_with_typed_command_argv():
    """An authorized approve press runs /email tg-approve <ref> <action>."""
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: uid == "111")
    query = _make_query("em:approve:tg-123:archive", user_id=111)

    calls = []

    def fake_handler(raw_args):
        calls.append(raw_args)
        return "tg-approve: state=EXECUTING"

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler", return_value=fake_handler
    ) as get_handler:
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    get_handler.assert_called_once_with("email")
    assert calls == ["tg-approve tg-123 archive"]
    query.answer.assert_awaited_once_with(text="tg-approve: state=EXECUTING")


@pytest.mark.asyncio
async def test_dismiss_press_dispatches_plugin_with_typed_command_argv():
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: uid == "111")
    query = _make_query("em:dismiss:tg-123", user_id=111)

    calls = []

    def fake_handler(raw_args):
        calls.append(raw_args)
        return "tg-dismiss: state=DISMISSED"

    with patch("hermes_cli.plugins.get_plugin_command_handler", return_value=fake_handler):
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    assert calls == ["tg-dismiss tg-123"]
    query.answer.assert_awaited_once_with(text="tg-dismiss: state=DISMISSED")


@pytest.mark.asyncio
async def test_unauthorized_press_answered_and_dropped_before_dispatch():
    """A valid payload from an unauthorized presser must never reach the plugin."""
    seen = []
    adapter = _make_adapter(
        callback_auth=lambda uid, **kw: seen.append((uid, kw)) or False
    )
    query = _make_query("em:approve:tg-123:archive", user_id=666)

    with patch("hermes_cli.plugins.get_plugin_command_handler") as get_handler:
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    get_handler.assert_not_called()
    query.answer.assert_awaited_once()
    # The same callback authorization check as ea:/gt: sees the full context.
    assert seen == [
        (
            "666",
            {
                "chat_id": -100,
                "chat_type": "group",
                "thread_id": None,
                "user_name": "Test",
            },
        )
    ]


@pytest.mark.asyncio
async def test_malformed_payload_answered_and_dropped_without_dispatch():
    """Fail closed: bad em: data is answered and dropped before auth or dispatch."""
    auth_calls = []
    adapter = _make_adapter(
        callback_auth=lambda uid, **_kw: auth_calls.append(uid) or True
    )
    query = _make_query("em:approve:tg-123:trash", user_id=111)

    with patch("hermes_cli.plugins.get_plugin_command_handler") as get_handler:
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    get_handler.assert_not_called()
    assert auth_calls == []
    query.answer.assert_awaited_once_with(text="Invalid email approval data.")


@pytest.mark.asyncio
async def test_missing_plugin_answered_without_crash():
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: True)
    query = _make_query("em:dismiss:tg-123", user_id=111)

    with patch("hermes_cli.plugins.get_plugin_command_handler", return_value=None):
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    query.answer.assert_awaited_once_with(
        text="❌ email-manager-operator plugin unavailable"
    )


@pytest.mark.asyncio
async def test_plugin_exception_answered_bounded():
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: True)
    query = _make_query("em:dismiss:tg-123", user_id=111)

    def exploding_handler(raw_args):
        raise RuntimeError("boom")

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler", return_value=exploding_handler
    ):
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    query.answer.assert_awaited_once_with(text="❌ email command failed")


@pytest.mark.asyncio
async def test_plugin_result_bounded_to_callback_answer_limit():
    """answerCallbackQuery caps text at 200 chars; the answer must fit."""
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: True)
    query = _make_query("em:dismiss:tg-123", user_id=111)

    with patch(
        "hermes_cli.plugins.get_plugin_command_handler",
        return_value=lambda raw_args: "x" * 5000,
    ):
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    query.answer.assert_awaited_once()
    assert len(query.answer.await_args.kwargs["text"]) <= 200


@pytest.mark.asyncio
async def test_async_plugin_handler_result_awaited():
    """Plugin command handlers may be coroutines, like the typed /email path."""
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: True)
    query = _make_query("em:dismiss:tg-123", user_id=111)

    async def async_handler(raw_args):
        await asyncio.sleep(0)
        return f"ran: {raw_args}"

    with patch("hermes_cli.plugins.get_plugin_command_handler", return_value=async_handler):
        await adapter._handle_callback_query(_make_update(query), SimpleNamespace())

    query.answer.assert_awaited_once_with(text="ran: tg-dismiss tg-123")
