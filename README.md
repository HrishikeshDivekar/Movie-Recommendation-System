# 🎬 Movie Recommendation System (Collaborative Filtering)

## 📌 Overview

This project implements a **Movie Recommendation System** using collaborative filtering techniques on a synthetic dataset. It demonstrates how to build, evaluate, and analyze recommendation models using matrix factorization and similarity-based approaches.

The notebook covers the full pipeline:

* Data generation
* Exploratory data analysis (EDA)
* Data transformation into sparse matrices
* Model training (NMF, SVD)
* Similarity computation
* Evaluation using regression metrics

The system is designed to simulate real-world recommendation engines like those used by streaming platforms.

---

## ⚙️ Technical Stack

* **Language:** Python
* **Libraries:**

  * `NumPy`, `Pandas` – data handling
  * `Matplotlib`, `Seaborn` – visualization
  * `SciPy` – sparse matrix operations
  * `Scikit-learn` – ML models & evaluation
* **Techniques Used:**

  * Collaborative Filtering
  * Matrix Factorization (NMF, SVD)
  * Cosine Similarity
  * Sparse Matrix Optimization

---

## 📊 Dataset

The dataset is **synthetically generated** to mimic real-world movie rating behavior.

### Key Characteristics:

* **Users:** 1000
* **Movies:** 500
* **Ratings:** ~47,574 (after deduplication) 
* **Rating Scale:** 1–5
* **Distribution Bias:** Higher probability for ratings 3–5

### Features:

* `user_id` – unique user identifier
* `movie_id` – unique movie identifier
* `rating` – user rating for a movie

---

## 🔍 Data Exploration

Basic statistical insights:

* **Average Rating:** ~3.35 
* **Standard Deviation:** ~1.24 
* **Sparsity:** ~90.49% 

High sparsity reflects real-world recommender systems where users rate only a small subset of items.

---

## 🏗️ System Architecture

### 1. Data Generation

* Random user–movie interactions
* Probabilistic rating distribution
* Duplicate removal for consistency

### 2. Data Transformation

* Convert dataset into **user-item matrix**
* Use **CSR (Compressed Sparse Row)** format for efficiency

### 3. Model Training

#### a. Matrix Factorization

* **NMF (Non-negative Matrix Factorization)**
* **Truncated SVD**

These methods decompose the user-item matrix into latent factors:

* User preferences
* Movie characteristics

#### b. Similarity-Based Filtering

* Compute **cosine similarity** between:

  * Users (user-based filtering)
  * Items (item-based filtering)

---

## 🧠 Recommendation Logic

### Approach 1: User-Based Collaborative Filtering

1. Find similar users
2. Recommend movies liked by similar users

### Approach 2: Item-Based Collaborative Filtering

1. Find similar movies
2. Recommend movies similar to those already rated

### Approach 3: Latent Factor Models

* Use NMF/SVD embeddings
* Predict missing ratings

---

## 📈 Model Evaluation

The system uses regression-based evaluation metrics:

* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**

These metrics evaluate how well predicted ratings match actual ratings.

---

## 🔄 Workflow

```
Data Generation → Cleaning → EDA → Matrix Conversion
        ↓
Model Training (NMF / SVD / Similarity)
        ↓
Prediction & Recommendation
        ↓
Evaluation (MSE, MAE)
```

---

## 🧩 Custom User Input (Extensibility)

The system can be extended to accept **real user input**:

### Input Options:

* New user ratings (e.g., rate 5–10 movies)
* User ID selection
* Movie preference filtering

### Example:

```python
new_user_ratings = {
    movie_id_1: 5,
    movie_id_2: 3,
    movie_id_3: 4
}
```

### Output:

* Top-N recommended movies
* Similar users/items
* Predicted ratings

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

2. Run the notebook:

```bash
jupyter notebook
```

3. Execute all cells step-by-step

---

## 📌 Key Features

* End-to-end recommendation pipeline
* Sparse matrix optimization
* Multiple recommendation strategies
* Scalable to real-world datasets
* Easily extendable for production systems

---

## ⚠️ Limitations

* Uses synthetic data (not real-world dataset)
* Cold-start problem not addressed
* No content-based filtering
* No real-time updates

---

## 🔮 Future Improvements

* Integrate real datasets (e.g., MovieLens)
* Add hybrid recommendation (content + collaborative)
* Implement deep learning models
* Build API / UI for deployment
* Handle cold-start users/items

---
