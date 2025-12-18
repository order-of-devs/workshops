import mlflow

from registry_prompt.flow_php_prompt import ASSISTANT_PROMPT_TEMPLATE
from registry_prompt.register_prompt import register_prompt, RegisterPrompt


def init_registry_prompt():
    register_prompt(
        RegisterPrompt(name="flow_php", prompt=ASSISTANT_PROMPT_TEMPLATE)
    )

def main():
    init_registry_prompt()


if __name__ == "__main__":
    main()
