from cachetools.func import lru_cache
from smolagents import CodeAgent, LiteLLMModel, FinalAnswerTool, InferenceClientModel, tool, ToolCallingAgent
from smolagents.remote_executors import DockerExecutor, ModalExecutor
model = LiteLLMModel("ollama/granite4", api_base="http://localhost:11434")

SYTEM_PROMPT = """
You are a helpful PHP programming assistant specialized in Flow PHP development. You provide clear, production-ready code with explanations. Always consider security, error handling, and best practices.
Example of dataframe:
```php
$df = new DataFrame([
    ['name', 'age'],
    ['Alice', 30],
    ['Bob', 25],
]);
```
If user ask about dataframe 
should use add keywords into query 
Example of Flow PHP code:
```php
$flow = new Flow();
$flow->from($df)
    ->map(function ($row) {
        return [
            'name' => $row['name'],
            'age' => $row['age'] + 1,
        ];
    })
    ->to($df);
```
"""
agent = ToolCallingAgent(tools=[
], model=model)
