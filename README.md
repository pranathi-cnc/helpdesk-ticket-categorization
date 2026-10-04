# Helpdesk Ticket Categorization

A helpdesk ticket classification project using Gemma and LoRA.

## Categories

* Billing
* Login
* Delivery

## Dataset

30 fictional tickets:

* 20 Training
* 5 Validation
* 5 Test

## Technologies

* Python
* Ollama
* Gemma 4
* Hugging Face Transformers
* PEFT
* LoRA
* PyTorch
* Google Colab

## Workflow

```text
Tickets
   ↓
Train / Validation / Test Split
   ↓
Gemma 4 Baseline
   ↓
LoRA Fine-Tuning
   ↓
Base vs LoRA Evaluation
```

## Results

| Model                   | Accuracy |
| ----------------------- | -------: |
| Ollama Gemma 4 Baseline |     100% |
| Gemma 3 1B Base         |      40% |
| Gemma 3 1B + LoRA       |      40% |

The LoRA experiment was performed on Google Colab using a Tesla T4 GPU.

## Project Structure

```text
data/        → Dataset files
outputs/     → Evaluation results
notebooks/   → Google Colab LoRA notebook
src/         → Python source code
README.md
requirements.txt
```

## Key Concepts

* Helpdesk ticket classification
* Supervised fine-tuning
* LoRA / PEFT
* Baseline evaluation
* Model comparison
