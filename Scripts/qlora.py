import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import (
    AlbertTokenizerFast,
    AlbertForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import evaluate
from peft import get_peft_model, LoraConfig, TaskType
from bitsandbytes.optim import AdamW8bit

# Load tokenizer and dataset
tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
dataset = load_dataset("squad")
squad_metric = evaluate.load("squad")

# Preprocess dataset
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    start_positions = []
    end_positions = []

    for i, offset_mapping in enumerate(inputs["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1

        start_pos = 0
        end_pos = 0
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= start_char < end:
                start_pos = idx
            if start < end_char <= end:
                end_pos = idx
                break

        if not (context_start <= start_pos <= context_end and context_start <= end_pos <= context_end):
            start_pos = 0
            end_pos = 0

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    inputs["start_positions"] = torch.tensor(start_positions)
    inputs["end_positions"] = torch.tensor(end_positions)
    return inputs

# Tokenize and format dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ["id", "title", "answers", "context", "question", "offset_mapping"]
)
tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Utility functions
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_training_time(trainer):
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    trainer.train()
    end.record()
    torch.cuda.synchronize()
    time_elapsed = start.elapsed_time(end) / 1000.0  # Convert ms to seconds
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
    return time_elapsed, max_memory

def compute_metrics(pred):
    predictions, labels = pred
    start_preds = torch.argmax(torch.tensor(predictions[0]), dim=-1).tolist()
    end_preds = torch.argmax(torch.tensor(predictions[1]), dim=-1).tolist()
    start_labels = labels[0].tolist()
    end_labels = labels[1].tolist()

    # Format predictions and references for squad_metric
    formatted_preds = []
    formatted_labels = []

    for i in range(len(start_preds)):
        # Decode predicted and actual spans
        pred_text = tokenizer.decode(
            train_dataset[i]["input_ids"][start_preds[i]:end_preds[i] + 1],
            skip_special_tokens=True
        )
        label_text = tokenizer.decode(
            train_dataset[i]["input_ids"][start_labels[i]:end_labels[i] + 1],
            skip_special_tokens=True
        )

        formatted_preds.append({"id": str(i), "prediction_text": pred_text})
        formatted_labels.append({"id": str(i), "answers": {"text": [label_text], "answer_start": [start_labels[i]]}})

    # Compute the metric
    result = squad_metric.compute(predictions=formatted_preds, references=formatted_labels)
    return result




# Plot Combined Loss



best_alpha = 32  # Replace with the best alpha
best_r = 8       # Replace with the best r
best_dropout = 0.1  # Replace with the best dropout


model_qlora = AlbertForQuestionAnswering.from_pretrained(
    "albert-base-v2",
    load_in_8bit=True  # Keep 8-bit quantization
)

# Define QLoRA Configuration
qlora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    inference_mode=False,
    r=best_r,
    lora_alpha=best_alpha,
    lora_dropout=best_dropout,
    target_modules=["query", "key", "value"]
)

# Apply QLoRA
model_qlora = get_peft_model(model_qlora, qlora_config)


training_args_qlora = TrainingArguments(
    output_dir="./results_qlora",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs_qlora",
    fp16=False # Ensure mixed precision is enabled
)

# Adjusting the optimizer for compatibility with FP16
optimizer_qlora = AdamW8bit(
    model_qlora.parameters(),
    lr=training_args_qlora.learning_rate
)

trainer_qlora = Trainer(
    model=model_qlora,
    args=training_args_qlora,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer_qlora, None),
    compute_metrics=compute_metrics,
)


# Train and Evaluate QLoRA Model
print("Training QLoRA model...")
time_qlora, memory_qlora = measure_training_time(trainer_qlora)
metrics_qlora = trainer_qlora.evaluate()
trainable_params_qlora = count_trainable_parameters(model_qlora)



# Report Results

print(f"Qlora Tuning: Params={trainable_params_qlora}, Time={time_qlora:.2f}s, "
      f"Memory={memory_qlora:.2f}MB, Metrics={metrics_qlora}")
