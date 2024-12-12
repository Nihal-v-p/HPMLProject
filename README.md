# Finetuning Comparison on ALBERT for Question Answering

## Project Description
This project aims to comprehensively compare full finetuning and sparse finetuning techniques (BitFit and LoRA) for the ALBERT (A Lite BERT) model on the SQuAD dataset. The primary focus is on evaluating parameter reduction, speedup, and memory footprint for both methods. We also explore whether sparse finetuning methods are effective for smaller models like ALBERT.

## Project Milestones and Status

| Milestone                                           | Status        |
|---------------------------------------------------- |---------------|
| Literature review on finetuning and sparse methods | Completed      |
| Data preprocessing and SQuAD integration           | Completed      |
| Implementation of full finetuning                  | Completed      |
| Implementation of BitFit                           | Completed      |
| Implementation of LoRA                             | Completed      |
| Comparative analysis of methods                    | Completed      |
| Visualization and reporting                        | Completed      |

## Repository and Code Structure

```
root/
|
|-- Scripts/
|   |-- main.py                 # Script for full finetuning,BitFit finetuning,LoRA finetuning
|   |-- qlora.py                # Script for qlora finetuning
|
|-- Results/                	# Graphs, Screenshots 
|
|-- README.md                   # Project overview (this file)
|-- requirements.txt            # Required Python packages
```

## Example Commands

1. Run main:
   ```bash
   python main.py
   or
   torchrun --nproc_per_node=<num_workers> main.py
   ```

2. Run QLoRA:
   ```bash
   python qlora.py
   ```

## Results and Observations

## Performance Comparison of Methods

| **Method** | **Params**    | **Exact-Match** | **F1 Score** |
|------------|---------------|-----------------|--------------|
| **Full**  | 11,094,530    | 67.87           | 52.84        |
| **BitFit**| 9,346         | 36.61           | 26.76        |
| **LoRA**  | 38,402        | 62.40           | 48.86        |
| **QLoRA** | 38,402        | 60.93           | 47.73        |

## Resource Comparison of Methods

| **Method** | **Memory Usage (MB)** | **Time (s)** |
|------------|-----------------------|--------------|
| **Full**  | 5177.95                | 2343.46      |
| **BitFit**| 4554.07                | 2063.60      |
| **LoRA**  | 5071.05                | 2299.06      |
| **QLoRA** | 3984.38                | 3324.77      |

### Observations
-LORA shows promising results, achieving close to full fine-tuning performance with significantly fewer parameters (about 0.3 percent)
-BitFit underperformed for this task, possibly due to the complexity of question answering
-Both sparse finetuning methods offer speed up compared to full fine-tuning
-QLoRA gives scores close to LoRA while using significantly less memory


---


