import mlflow
from pydantic import BaseModel

class RegisterPrompt(BaseModel):
    name: str
    prompt: str
    commit_message: str = "Initial commit"
    tags: dict[str, str] = {
        "author": "John Doe"
    }

def register_prompt(prompt: RegisterPrompt):
    mlflow.set_registry_uri("http://192.168.97.4:5000")
    mlflow.genai.register_prompt(
        name=prompt.name,
        template=prompt.prompt,
        commit_message=prompt.commit_message,
        tags=prompt.tags
    )