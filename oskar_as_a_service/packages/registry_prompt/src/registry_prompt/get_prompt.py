import mlflow


def get_prompt(name: str = "flow_php"):
    mlflow.set_registry_uri("http://192.168.97.4:5000")
    return mlflow.genai.load_prompt("flow_php")