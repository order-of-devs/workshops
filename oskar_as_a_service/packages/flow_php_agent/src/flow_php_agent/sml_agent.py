from cachetools.func import lru_cache
from smolagents import CodeAgent, LiteLLMModel, FinalAnswerTool, InferenceClientModel, tool, ToolCallingAgent
from smolagents.remote_executors import DockerExecutor, ModalExecutor
model = LiteLLMModel("ollama/granite4", api_base="http://localhost:11434")

agent = ToolCallingAgent(tools=[
], model=model)
