# IMDB Movie Review Sentiment Analysis with an RNN

A PyTorch Recurrent Neural Network that reads a movie review and predicts whether it's **positive** or **negative**.

## Results

| Metric | Score |
|---|---|
| Test accuracy | **85.16%** |
| Test reviews | 9,917 |
| Trainable parameters | ~657K |

## Dataset

**[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**

| | Count |
|---|---|
| Total reviews | 50,000 |
| After removing duplicates | 49,582 |
| Positive | 24,884 |
| Negative | 24,698 |

## Pipeline

### 1. Text preprocessing

| Step | What it does | Tool |
|---|---|---|
| Lowercase | `"Great"` → `"great"` | `str.lower()` |
| Remove URLs | strips `http…` links | `re.sub` |
| Remove HTML | strips tags like `<br />` | `re.sub` |
| Remove punctuation | keeps only letters, digits and spaces | `re.sub` |
| Remove stopwords | drops common words (*the, is, and…*) | NLTK |
| Stemming | `"filming"` → `"film"` | NLTK `PorterStemmer` |

### 2. Encoding and vectorisation

- **Labels:** `LabelEncoder` maps negative to `0` and positive to `1`
- **Text:** `TfidfVectorizer(max_features=5000)` turns each review into a 5,000-number vector that scores how important each word is

### 3. Train / test split

80 / 20 split (`random_state=42`): **39,665** training and **9,917** test reviews, loaded in batches of 64.

## Model Architecture

```
TF-IDF vector (5000)
  → RNN layer (128 hidden units, tanh)
  → Linear (128 → 1)
  → Sigmoid → probability of "positive"
```

## Training

| Setting | Value |
|---|---|
| Loss | `BCELoss` (binary cross-entropy) |
| Output activation | Sigmoid |
| Optimizer | Adam |
| Epochs | 10 |
| Batch size | 64 |
| Decision rule | probability > 0.5 → positive |

## Tech Stack

Python · PyTorch · NLTK · scikit-learn · pandas · Jupyter
