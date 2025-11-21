import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

prompt_path = Path(os.path.abspath(__file__), "..", "..", "prompts", "dsl.json")

prompt_path = "/Users/mkubaszek/Projects/ood/workshops/oskar_as_a_service/prompts/dsl.json"


def load_dataset():
    c = JSONLoader(file_path=prompt_path, jq_schema=".", text_content=False)
    return c.load()


def initial_rag():
    return Chroma(
        collection_name="flowphp-dsl",
        persist_directory="./chroma_db/flowphp_dsl",
    )


def add_dataset_to_rag(rag: Chroma, dataset):
    rag.add_documents(dataset)


def find_similar_documents(rag: Chroma, query: str, k: int = 3) -> list[Document]:
    return rag.similarity_search(query, k=k)


def concatenate_documents(documents: list[Document]) -> str:
    return "\n".join([doc.page_content for doc in documents])


def load_llm():
    return ChatOllama(model="qwen3:4b", base_url="http://localhost:11434")


prompt_temlate = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""
    Context: {context}
    User Input: {user_input}
""",
)


def build_prompt(context: str, user_query) -> str:
    return prompt_temlate.format(context=context, user_input=user_query)


def invoke_response(llm: ChatOllama, rag: Chroma, user_query: str):
    context = concatenate_documents(find_similar_documents(rag, user_query))
    return llm.invoke([HumanMessage(content=build_prompt(context, user_query))])
