from unittest.mock import MagicMock, patch
import httpx
import pytest
from google.genai import errors
import client
from gemini_client import get_gemini_response

# Убираем реальные паузы между повторами tenacity, чтобы тесты были быстрыми.
get_gemini_response.retry.wait = lambda retry_state: 0


def _mock_generate(side_effects):
    return patch.object(
        client.client.models, "generate_content", side_effect=side_effects
    )


def test_retries_on_rate_limit_then_succeeds():
    fake_response = MagicMock(text="ok")
    with _mock_generate(
        [errors.ClientError(429, {"error": {"message": "rate limited"}}), fake_response]
    ) as mocked:
        result = get_gemini_response("промпт")

    assert result == "ok"
    assert mocked.call_count == 2


def test_retries_on_server_error():
    fake_response = MagicMock(text="ok")
    with _mock_generate(
        [errors.ServerError(500, {"error": {"message": "boom"}}), fake_response]
    ) as mocked:
        result = get_gemini_response("промпт")

    assert result == "ok"
    assert mocked.call_count == 2


def test_retries_on_timeout():
    fake_response = MagicMock(text="ok")
    with _mock_generate([httpx.ReadTimeout("timed out"), fake_response]) as mocked:
        result = get_gemini_response("промпт")

    assert result == "ok"
    assert mocked.call_count == 2


def test_gives_up_after_three_retries():
    errs = [errors.ServerError(500, {"error": {"message": "boom"}})] * 10
    with _mock_generate(errs) as mocked:
        with pytest.raises(errors.ServerError):
            get_gemini_response("промпт")

    assert mocked.call_count == 4


def test_forbidden_is_not_retried():
    with _mock_generate(
        [errors.ClientError(403, {"error": {"message": "bad api key"}})]
    ) as mocked:
        with pytest.raises(errors.ClientError):
            get_gemini_response("промпт")

    assert mocked.call_count == 1


def test_sleeps_before_each_request_for_rate_limiting():
    fake_response = MagicMock(text="ok")
    with _mock_generate([fake_response]):
        with patch.object(client.time, "sleep") as mocked_sleep:
            get_gemini_response("промпт")

    mocked_sleep.assert_called_once_with(client.REQUEST_DELAY_SECONDS)