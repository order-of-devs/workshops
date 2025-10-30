import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pydantic import BaseModel
    class Item(BaseModel):
        pass
    class ModelResult(BaseModel):
        pass
    return (BaseModel,)


@app.cell
def _(BaseModel):
    class Prompt(BaseModel):
        title: str
        content: str
    class PromptsFile(BaseModel):
        prompts: list[Prompt]
    return (PromptsFile,)


@app.cell
def _():
    from pathlib import Path
    sourcing_path = Path("", "prompts", "emmet_sourcing.json")
    sourcing_path
    return Path, sourcing_path


@app.cell
def _(PromptsFile, sourcing_path):
    prompt_file = PromptsFile.parse_file(sourcing_path)
    return (prompt_file,)


@app.cell
def _(prompt_file):
    prompts = prompt_file.prompts
    prompts
    return (prompts,)


@app.cell
def _(prompts):
    from transformers import pipeline


    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-1.5B-Instruct")

    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt.content},
        ]
        result = pipe(messages)
    return (result,)


@app.cell
def _(result):
    result
    return


@app.cell
def _(result):
    content = result[0]["generated_text"][1]["content"]
    return (content,)


@app.cell
def _(Path, content):
    result_path = Path("", "reports", "output.txt")
    with open(result_path, "w+") as f:
        f.write(content)
    return


if __name__ == "__main__":
    app.run()
