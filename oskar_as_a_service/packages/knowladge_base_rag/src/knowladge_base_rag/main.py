from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader, GitLoader
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

root_path = Path(__file__).resolve().parents[4]
dataset_path = root_path / "prompts" / "dsl.json"
rag_path =root_path / "chroma_db"
repo_path = root_path / "repositories"

def filter_php_file(path: str) -> bool:
    return path.endswith(".ts")

# def load_json_dataset():
#     loader = JSONLoader(file_path=dataset_path, jq_schema=".", text_content=False)
#     return loader.load()

def load_github_dataset(repo_url: str):
    loader = GitLoader(
        clone_url=repo_url,
        branch="main",
        repo_path=str(repo_path / repo_url.split("/")[-1]),
        file_filter=filter_php_file,
    )
    return loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter.from_language(
            chunk_size=1000, chunk_overlap=100,
            language=Language.TS
        )
    )


def initial_rag(documents: list[Document]):
    return Chroma.from_documents(
        documents=documents,
        embedding=GPT4AllEmbeddings(),
        collection_name="emmet",
        persist_directory=str(rag_path / "emmet"),
    )


def main():
    documents = []
    documents.extend(load_github_dataset("https://github.com/event-driven-io/emmett"))
    documents.extend(load_github_dataset("https://github.com/oskardudycz/EventSourcing.NodeJS"))
    initial_rag([
        doc for doc in documents if doc.page_content.strip() != ""
    ])


if __name__ == "__main__":
    main()