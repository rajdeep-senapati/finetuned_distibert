# Fine-Tuned DistilBERT for Text Classification

This project demonstrates how to fine-tune the DistilBERT model for a custom text classification task using the Hugging Face Transformers library. The workflow includes preprocessing, tokenization, model training, evaluation, and prediction.

## 🔍 Project Overview

- **Model Used**: DistilBERT (`distilbert-base-uncased`)
- **Task**: Text Classification
- **Libraries**: Hugging Face Transformers, Datasets, PyTorch, Sklearn
- **Notebook**: `finetuned_distilbert.ipynb`

## 📁 File Structure

```
finetuned_distilbert/
├── finetuned_distilbert.ipynb   # Jupyter Notebook with all steps
├── dataset/                     # (optional) Custom dataset used
└── README.md                    # Project description
```

## ✅ Features

- Tokenization using DistilBERT tokenizer
- Dataset loading and preprocessing
- Model fine-tuning on labeled data
- Training with evaluation at each step
- Performance metrics like accuracy, precision, recall, F1-score
- Prediction on test samples

## 🛠️ How to Run

1. Clone this repository or download the notebook.
2. Install required libraries:
   ```bash
   pip install transformers datasets torch scikit-learn
   ```
3. Open the notebook:
   ```bash
   jupyter notebook finetuned_distilbert.ipynb
   ```
4. Follow along and run each cell in order.

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

These metrics help evaluate the performance of the fine-tuned model on unseen data.

## 📌 Highlights

- Lightweight and faster than BERT
- Ideal for resource-constrained environments
- Easy to integrate with downstream applications

## 🧠 Future Improvements

- Use `Trainer` API for more modular training loops
- Integrate hyperparameter tuning using Optuna or Ray Tune
- Deploy as an API using FastAPI or Flask

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

---

*Developed with 💻 by Rajdeep Senapati*
