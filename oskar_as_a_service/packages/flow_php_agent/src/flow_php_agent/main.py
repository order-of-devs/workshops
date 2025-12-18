from pathlib import Path

import gradio as gr
from cachetools.func import lru_cache
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings

from flow_php_agent.sml_agent import agent
from registry_prompt.get_prompt import get_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAG_PATH = PROJECT_ROOT / "chroma_db" / "flowphp"
REPOSITORY_PATH = PROJECT_ROOT / "repositories" / "flowphp"



@lru_cache
def get_rag():
    return Chroma(
        embedding_function=GPT4AllEmbeddings(),
        collection_name="doc",
        persist_directory=str(RAG_PATH),
    )

def format_prompt(
        message: str,
        context: str,
        history: list[tuple[str, str]]
) -> str:
    return get_prompt().template.format(
        max_tokens=1024,
        message=message,
        context=context,
        history="\n".join([f"{role}: {content}" for role, content in history])
    )

def build_context(documents: list[Document]) -> str:
    prepared_context = []
    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        header = metadata.get("header") or f"Document page={i}"
        prepared_context.append(f"**{header}.** {doc.page_content}")
    return "\n".join(prepared_context)

def generate_llm_response(_agent, prompt: str):
    return _agent.run(prompt)

def handle_message(message, history):
    if not message or message.strip() == "":
        return "Please enter a non-empty message."
    rag = get_rag()
    docs = rag.similarity_search(message, k=5)
    print(f"Found {len(docs)} documents for query: {message}")
    prompt = format_prompt(message, context=build_context(docs), history=history)
    print(f"Prompt: {prompt}")
    response = generate_llm_response(agent, prompt)
    print(f"Response: {response}")
    return response

def create_chat_ui():
    with gr.Blocks(fill_height=True) as block:
        gr.ChatInterface(
            handle_message,
            title="Chat",
            description="This is a chat interface for Emmet",
        )
    return block
def main():
    chat = create_chat_ui()
    chat.launch(pwa=True)

if __name__ == "__main__":
    main()
