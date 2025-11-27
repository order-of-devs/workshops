from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader, GitLoader
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BRANCH = "1.x"

#dataset_path = root_path / "prompts" / "dsl.json"
RAG_PATH = PROJECT_ROOT / "chroma_db" / "flowphp"
REPOSITORY_PATH = PROJECT_ROOT / "repositories" / "flowphp"

# def filter_file(path: str, ends_with: str) -> bool:
#     return path.endswith(ends_with)

def filter_by_names(path: str, names: list[str], endswith: str | None = None) -> bool:
    return any(name in path for name in names) and (endswith is None or path.endswith(endswith))

# def load_json_dataset():
#     loader = JSONLoader(file_path=dataset_path, jq_schema=".", text_content=False)
#     return loader.load()

def load_github_markdown_dataset(repo_url: str):
    loader = GitLoader(
        clone_url=repo_url,
        branch=BRANCH,
        repo_path=str(PROJECT_ROOT / repo_url.split("/")[-1]),
        file_filter=lambda path: filter_by_names(path, ['topics', 'resources'], endswith=".md"),
    )
    return loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter.from_language(
            chunk_size=1000, chunk_overlap=100,
            language=Language.MARKDOWN
        )
    )


def initial_rag(documents: list[Document]):
    return Chroma.from_documents(
        documents=documents,
        embedding=GPT4AllEmbeddings(),
        collection_name="doc",
        persist_directory=str(RAG_PATH),
    )


def main():
    documents = []
    documents.extend(load_github_markdown_dataset("https://github.com/flow-php/flow"))
    initial_rag([
        doc for doc in documents if doc.page_content.strip() != ""
    ])


if __name__ == "__main__":
    main()