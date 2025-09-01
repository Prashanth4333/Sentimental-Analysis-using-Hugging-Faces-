# 📊 Sentiment Analysis with Hugging Face 🤗

This project demonstrates how to perform Sentiment Analysis using Hugging Face Transformers and Datasets.
The model is fine-tuned on the IMDb dataset and trained in Google Colab using the Trainer API.


# 🚀 Features

Uses Hugging Face Transformers for model training and inference.

Fine-tunes a pretrained model (BERT/RoBERTa) for sentiment classification.

Works directly in Google Colab (no local setup required).

Evaluates model with accuracy and loss metrics.

Example predictions on new text inputs.


# 🛠️ Tech Stack

Python

Hugging Face Transformers

Hugging Face Datasets

Google Colab (GPU acceleration)


# 📂 Project Workflow

Import Libraries

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


Load Dataset

dataset = load_dataset("imdb")


Preprocess Data

Tokenization with Hugging Face tokenizer

Split into train and test sets

Training Setup

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
)


Train the Model

trainer.train()


Evaluate

results = trainer.evaluate()
print("Evaluation Results:", results)


Make Predictions

inputs = tokenizer("This movie was amazing!", return_tensors="pt")
outputs = model(**inputs)


# 📊 Sample Results

After 2 epochs of training:

Epoch	Training Loss	Validation Loss	Accuracy
1	0.3088	0.4323	81.2%
2	0.1098	0.6947	83.2%


# ▶️ Run on Google Colab

You can open and run this project directly in Google Colab:


# 📌 Future Improvements

Try RoBERTa or DistilBERT for faster training.

Add more epochs for better accuracy.

Experiment with different datasets like SST-2.


# 🤝 Acknowledgments

Hugging Face Transformers

Hugging Face Datasets

Google Colab for free GPU
