from pathlib import Path

from cachetools.func import lru_cache
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from pydantic_ai import Tool, FunctionToolset, RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAG_PATH = PROJECT_ROOT / "chroma_db" / "flowphp"


@lru_cache
def get_rag():
    return Chroma(
        embedding_function=GPT4AllEmbeddings(),
        collection_name="doc",
        persist_directory=str(RAG_PATH),
    )

def build_context(documents: list[Document]) -> str:
    prepared_context = []
    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        header = metadata.get("header") or f"Document page={i}"
        prepared_context.append(f"**{header}.** {doc.page_content}")
    return "\n".join(prepared_context)


def find_similar_documents(message: str):
    docs = get_rag().similarity_search(message, k=5)
    return build_context(docs)

search_toolset = Tool(find_similar_documents)