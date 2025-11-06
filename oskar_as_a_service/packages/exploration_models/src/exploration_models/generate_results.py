from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from transformers import pipeline

Model = str
Models = list[Model]

MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "google/gemma-3-4b-it", "ibm-granite/granite-4.0-h-1b"]
PipelineModel = namedtuple("PipelineModel", ["pipe", "name"])
ResponseModel = namedtuple("ResponseModel", ["response", "name"])


def prepare_models(models: Models) -> list[Any]:
    pipes = []

    for model in models:
        pipe = pipeline("text-generation", model=model)
        pipes.append(PipelineModel(pipe, model))
    return pipes


def generate_response(pipe_model: PipelineModel, prompt: str) -> ResponseModel:
    messages = [
        {"role": "user", "content": prompt},
    ]
    result = pipe_model.pipe(messages)
    content = result[0]["generated_text"][1]["content"]
    return ResponseModel(content, pipe_model.name)


def display_results(results: list[ResponseModel]):
    for result in results:
        print(result.response)
        print("----\n")


def generate_result():
    models = prepare_models(MODELS)
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                generate_response,
                model,
                "Generate an empty class in python. Result print as markdown",
            )
            for model in models
        ]
        for future in as_completed(futures):
            results.append(future.result())
    display_results(results)
    return results


generate_result()
