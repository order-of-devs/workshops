from pathlib import Path

import gradio as gr
from cachetools.func import lru_cache
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAG_PATH = PROJECT_ROOT / "chroma_db" / "flowphp"
REPOSITORY_PATH = PROJECT_ROOT / "repositories" / "flowphp"

ASSISTANT_PROMPT_TEMPLATE = """
You are a helpful assistant that can answer questions about Flow PHP codebase.
## Core Instructions
1. ALWAYS prioritize the provided context as your primary source of truth
2. Keep responses concise and directly relevant to the user's question
3. Maximum {max_tokens} tokens per response
4. Use conversation history to maintain context and avoid repetition

## Conversation History
{history}

## Context Usage
- Code context: {context}
- User message: {message}

When the context contains relevant information:
- Use it to answer accurately and completely
- Try do describe information with detailes

When the context lacks relevant information:
- Clearly state: "I don't see this in the available codebase context"
- Suggest what to search for or where to look
- Do NOT make assumptions or use general knowledge to fill gaps

For explanation queries:
- Reference specific files/functions from context

For architecture/structure queries:
- List relevant files, classes, or modules from context
- Show relationships or hierarchies when present
- Use bullet points for clarity

## Tone
- Professional and helpful
- Direct and specific
- Avoid unnecessary pleasantries or repetition"""

@lru_cache
def get_rag():
    return Chroma(
        embedding_function=GPT4AllEmbeddings(),
        collection_name="doc",
        persist_directory=str(RAG_PATH),
    )

@lru_cache
def get_llm():
    return ChatOllama(
        model="granite4",
        temperature=0.0,
        base_url="http://localhost:11434")

def format_prompt(
        message: str,
        context: str,
        history: list[tuple[str, str]]
) -> str:
    return ASSISTANT_PROMPT_TEMPLATE.format(
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

def generate_llm_response(llm, prompt: str):
    return llm.invoke([HumanMessage(content=prompt)]).content

def handle_message(message, history):
    if not message or message.strip() == "":
        return "Please enter a non-empty message."
    rag = get_rag()
    llm = get_llm()
    docs = rag.similarity_search(message, k=5)

    prompt = format_prompt(message, context=build_context(docs), history=history)

    response = generate_llm_response(llm, prompt)
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
