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




def plot_combined_loss(trainers, labels, filename):
    plt.figure(figsize=(10, 8))

    for trainer, label in zip(trainers, labels):
        logs = trainer.state.log_history
        losses = [log["loss"] for log in logs if "loss" in log]
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Fine-Tuning Methods")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Full Fine-Tuning
print("Starting Full Fine-Tuning...")
model_full = AlbertForQuestionAnswering.from_pretrained("albert-base-v2")

training_args_full = TrainingArguments(
    output_dir="./results_full",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs_full",
    fp16=True,
    dataloader_drop_last=True
)

trainer_full = Trainer(
    model=model_full,
    args=training_args_full,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

time_full, memory_full = measure_training_time(trainer_full)
metrics_full = trainer_full.evaluate()
trainable_params_full = count_trainable_parameters(model_full)

print(f"Full Fine-Tuning: Params={trainable_params_full}, Time={time_full:.2f}s, "
      f"Memory={memory_full:.2f}MB, Metrics={metrics_full}")

# BitFit
print("Starting BitFit...")
model_bitfit = AlbertForQuestionAnswering.from_pretrained("albert-base-v2")
for name, param in model_bitfit.named_parameters():
    if "bias" not in name:
        param.requires_grad = False

training_args_bitfit = TrainingArguments(
    output_dir="./results_bitfit",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs_bitfit",
    fp16=True,
    dataloader_drop_last=True
)

trainer_bitfit = Trainer(
    model=model_bitfit,
    args=training_args_bitfit,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

time_bitfit, memory_bitfit = measure_training_time(trainer_bitfit)
metrics_bitfit = trainer_bitfit.evaluate()
trainable_params_bitfit = count_trainable_parameters(model_bitfit)

print(f"BitFit: Params={trainable_params_bitfit}, Time={time_bitfit:.2f}s, "
      f"Memory={memory_bitfit:.2f}MB, Metrics={metrics_bitfit}")

# LoRA
print("Starting LoRA...")
model_lora = AlbertForQuestionAnswering.from_pretrained("albert-base-v2")
lora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)
model_lora = get_peft_model(model_lora, lora_config)

training_args_lora = TrainingArguments(
    output_dir="./results_lora",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs_lora",
    fp16=True,
    dataloader_drop_last=True
)

trainer_lora = Trainer(
    model=model_lora,
    args=training_args_lora,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

time_lora, memory_lora = measure_training_time(trainer_lora)
metrics_lora = trainer_lora.evaluate()
trainable_params_lora = count_trainable_parameters(model_lora)

print(f"LoRA: Params={trainable_params_lora}, Time={time_lora:.2f}s, "
      f"Memory={memory_lora:.2f}MB, Metrics={metrics_lora}")



plot_combined_loss(
    trainers=[trainer_full, trainer_bitfit, trainer_lora],
    labels=["Full Fine-Tuning", "BitFit", "LoRA"],
    filename="combined_loss.png"
)

# Report Results
print("\nResults Summary:")
print(f"Full Fine-Tuning: Params={trainable_params_full}, Time={time_full:.2f}s, "
      f"Memory={memory_full:.2f}MB, Metrics={metrics_full}")
print(f"BitFit: Params={trainable_params_bitfit}, Time={time_bitfit:.2f}s, "
      f"Memory={memory_bitfit:.2f}MB, Metrics={metrics_bitfit}")
print(f"LoRA: Params={trainable_params_lora}, Time={time_lora:.2f}s, "
      f"Memory={memory_lora:.2f}MB, Metrics={metrics_lora}")
