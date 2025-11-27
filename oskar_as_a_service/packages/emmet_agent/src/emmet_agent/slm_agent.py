from typing import Sequence

from cachetools.func import lru_cache
from pydantic_ai import Agent, Tool
from emmet_agent.prompts import SYSTEM_PROMPT_TEMPLATE
type Model = str
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
class Task:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class SlmAgent:
    def __init__(self,
                 system_prompt: str,
                 model_name: str = "granite4",
                 tools: Sequence[Tool] = ()
                 ):
        provider = OllamaProvider(
            base_url="http://localhost:11434/v1/",
        )
        model = OpenAIChatModel(
            provider=provider,
            model_name=model_name,
        )
        self._agent = Agent(
            system_prompt=system_prompt,
            model=model,
            tools=tools,
        )

    def run(self, message: str):
        result = self._agent.run_sync(message)
        return result.output


def get_slm_agent(tools: Sequence[Tool] = ()):
    return SlmAgent(
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        model_name="granite4",
        tools=tools,
    )