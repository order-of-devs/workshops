from pathlib import Path

import torch
from datasets import load_dataset
from evaluate import evaluator
from trl import SFTTrainer
from peft import LoraConfig

path = Path(__file__).parent / "flow-php-dataset.jsonl"
flow_dataset = load_dataset("json", data_files=str(path), split="train")
flow_dataset_split = flow_dataset.train_test_split(test_size=0.3, shuffle=True, seed=42)

trainer = SFTTrainer(
    model="ibm-granite/granite-4.0-h-350m",
    train_dataset=flow_dataset_split["train"],
    peft_config=LoraConfig(
        target_modules=["q_proj", "v_proj"],
    )
)


trainer.train()

our_evaluator = evaluator("question-answering")
result = our_evaluator.compute(
    model_or_pipeline=trainer.model,
    data=flow_dataset_split["test"],
)
print(result)
if __name__ == "__main__":
    print(flow_dataset)


