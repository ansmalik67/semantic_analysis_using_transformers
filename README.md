# Sentiment Analysis with Transformers

This project demonstrates how to use a transformer model (DistilBERT) to perform sentiment analysis on movie reviews from the IMDb dataset.

---

## What It Does
- Classifies movie reviews as **positive** or **negative**
- Uses a pre-trained **DistilBERT** model
- Fine-tunes on IMDb dataset
- Evaluates model accuracy and performance
- Visualizes training and validation results
- Saves the model for later use

---

## How to Use

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/sentiment-transformer.git
cd sentiment-transformer
```

2. **Install Requirements**
```bash
pip install transformers datasets scikit-learn matplotlib
```

3. **Run the Training Notebook**
- Open `transformer_training_visualization.ipynb`
- Run all cells to train and evaluate the model

4. **Use the Trained Model for Predictions**
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./model_output", tokenizer="./model_output")
print(classifier("The movie was amazing!"))
```

---

## Results
- **Accuracy:** ~93%
- Model shows good performance on test data

---

## Files
- `transformer_training_visualization.ipynb`: Full training process
- `transformer_sentiment_analysis.ipynb`: Simple predictions with pipeline
- `model_output/`: Saved model and tokenizer

---

## Credits
- Built using [Hugging Face Transformers](https://huggingface.co/transformers/)
- Dataset: [IMDb](https://huggingface.co/datasets/imdb)

---

© 2025 – For educational use.