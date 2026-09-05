import logging
import os
import time
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10
REQUEST_DELAY_SECONDS = 0.3


client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000),
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError) and exc.code == 429:  # rate limit
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def get_gemini_response(prompt: str) -> str:
    time.sleep(REQUEST_DELAY_SECONDS)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
    )
    return response.text


def main() -> None:
    try:
        answer = get_gemini_response("Whats rate limits?")
        print(answer)
    except httpx.TimeoutException:
        logger.error("Запрос к Gemini превысил таймаут (%s секунд).", TIMEOUT_SECONDS)
    except errors.ClientError as exc:
        logger.error("Ошибка клиента (%s): %s", exc.code, exc.message)
    except errors.ServerError as exc:
        logger.error("Сервер Gemini не смог обработать запрос (%s): %s", exc.code, exc.message)
