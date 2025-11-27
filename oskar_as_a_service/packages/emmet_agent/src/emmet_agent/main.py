
import gradio as gr

from emmet_agent.slm_agent import get_slm_agent
from emmet_agent.rag import search_toolset


def handle_message(message, history):
    if not message or message.strip() == "":
        return "Please enter a non-empty message."
    slm_agent = get_slm_agent([search_toolset])
    return slm_agent.run(message)

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
