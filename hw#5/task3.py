import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

QA_PROMPT = ChatPromptTemplate.from_template(
    "Ответь на вопрос, используя только приведённый ниже контекст. "
    "Если в контексте нет ответа, честно скажи, что не знаешь.\n\n"
    "Контекст:\n{context}\n\nВопрос: {input}"
)


def build_qa_chain(file_path: str, llm=None, embeddings=None):
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key)
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    docs = TextLoader(file_path, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)


def ask(file_path: str, question: str, llm=None, embeddings=None) -> str:
    qa_chain = build_qa_chain(file_path, llm=llm, embeddings=embeddings)
    result = qa_chain.invoke({"input": question})
    return result["answer"]


def main() -> None:
    file_path = "sample_document.txt"
    question = "Как LangChain и PromptHub дополняют друг друга?"

    print(f"Файл: {file_path}")
    print(f"Вопрос: {question}\n")
    print(ask(file_path, question))


if __name__ == "__main__":
    main()