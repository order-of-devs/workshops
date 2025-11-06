from deepeval import compare
from deepeval.metrics import ArenaGEval
from deepeval.models import OllamaModel
from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams

import tests.generate_results as generate_results

results = generate_results.generate_result()

test_case = ArenaTestCase(
    contestants={
        results[0].name: LLMTestCase(
            input="Generate an empty class in python. Result print as markdown",
            expected_output="```python\nclass EmptyClass:\n    pass\n```",
            actual_output=results[0].response,
        ),
        results[1].name: LLMTestCase(
            input="Generate an empty class in python. Result print as markdown",
            expected_output="```python\nclass EmptyClass:\n    pass\n```",
            actual_output=results[1].response,
        ),
        results[2].name: LLMTestCase(
            input="Generate an empty class in python. Result print as markdown",
            expected_output="```python\nclass EmptyClass:\n    pass\n```",
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
