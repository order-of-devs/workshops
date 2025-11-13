from deepeval import compare
from deepeval.metrics import ArenaGEval
from deepeval.models import OllamaModel
from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams
from exploration_models.generator_results import generate_results

# Chcemy dobrą bazę pod nasze dane
prompt = "Generate an basic datafame in python using pandas with 3 columns: Name, Age, City and 3 rows of data. Result print as markdown"
expected_output = "```python\nimport pandas as pd\n\ndata = {\n    'Name': ['Alice', 'Bob', 'Charlie'],\n    'Age': [25, 30, 35],\n    'City': ['New York', 'Los Angeles', 'Chicago']\n}\n\ndf = pd.DataFrame(data)\nprint(df)\n```"
results = generate_results(prompt)

test_case = ArenaTestCase(
    contestants={
        results[0].name: LLMTestCase(
            input=prompt,
            expected_output=expected_output,
            actual_output=results[0].response,
        ),
        results[1].name: LLMTestCase(
            input=prompt,
            expected_output=expected_output,
            actual_output=results[1].response,
        ),
        results[2].name: LLMTestCase(
            input=prompt,
            expected_output=expected_output,
            actual_output=results[2].response,
        ),
    },
)

arena_phi = ArenaGEval(
    name="Friendly",
    criteria="Choose the winner of the more friendly contestant based on the input and actual output",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model=OllamaModel(model="deepseek-r1:8b"),
)

result = compare(test_cases=[test_case], metric=arena_phi)
print(result)
