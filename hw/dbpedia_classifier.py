import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

# Step 1: Load dataset
dataset = load_dataset("fancyzhx/dbpedia_14")

# Step 2: Tokenize
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess(example):
    return tokenizer(example["content"], truncation=True)

tokenized = dataset.map(preprocess, batched=True)

# Step 3: Model setup
num_labels = 14
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

# Optional: Weight balancing if class imbalance is large (here we skip it)

# Step 4: Evaluation metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Step 5: TrainingArguments
training_args = TrainingArguments(
    output_dir="bert-dbpedia",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"].select(range(500)),  # for early eval
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# Step 7: Train
trainer.train()

# Step 8: Evaluate on first 500 test samples
test_subset = tokenized["test"].select(range(500))
predictions = trainer.predict(test_subset)
print("Accuracy on first 500 test samples:", predictions.metrics["test_accuracy"])

# Optional: Save screenshot of output
with open("accuracy.txt", "w") as f:
    f.write(f"Accuracy on first 500 test samples: {predictions.metrics['test_accuracy']:.4f}\n")
