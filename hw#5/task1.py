"""
Домашнее задание, п.1: простая цепочка LangChain для суммаризации текста
со страницы по URL.

Написано по образцу lesson-ai-05/ai5-1.py из репозитория курса — тот же
набор компонентов (WebBaseLoader + ChatGoogleGenerativeAI +
create_stuff_documents_chain), только оформлено как переиспользуемая
функция с произвольным URL, а не жёстко зашитым в код.
"""

import os
import sys

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Напишите краткое изложение следующего текста на русском языке "
    "(3-5 предложений):\n\n{context}"
)


def summarize_url(url: str, llm=None) -> str:
    """
    Загружает страницу по `url` и возвращает её краткое содержание.

    :param url: адрес веб-страницы, которую нужно суммаризировать.
    :param llm: языковая модель LangChain. По умолчанию — Gemini
        (модель из GEMINI_CHAT_MODEL). Параметр удобен для тестов —
        можно подставить фейковую модель вместо реального API.
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key)

    docs = WebBaseLoader(url).load()

    chain = create_stuff_documents_chain(llm, SUMMARY_PROMPT)

    return chain.invoke({"context": docs})


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://habr.com/ru/articles/883604/"
    print(f"Суммаризирую: {url}\n")
    print(summarize_url(url))


if __name__ == "__main__":
    main()