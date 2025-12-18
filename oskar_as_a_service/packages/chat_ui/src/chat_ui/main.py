from pathlib import Path

import gradio as gr
from cachetools.func import lru_cache
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
root_path = Path(__file__).resolve().parents[4]
rag_path =root_path / "chroma_db"

@lru_cache
def get_rag():
    return Chroma(
        embedding_function=GPT4AllEmbeddings(),
        collection_name="emmet",
        persist_directory=str(rag_path / "emmet"),
    )
@lru_cache
def get_llm():
    return ChatOllama(
        model="granite4",
        temperature=0.0,
        base_url="http://localhost:11434")

prompt = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""
    Context: {context}
    User Input: {user_input}
    """
)
def handle_message(message, history):
    rag = get_rag()
    docs = rag.similarity_search(message, k=5)
    r =prompt.format(context=docs, user_input=message)
    response = get_llm().invoke([HumanMessage(content=r)])
    return response.content

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
