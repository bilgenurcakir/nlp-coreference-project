

# 📄 Turkish Coreference Resolution System  
## Statistical Machine Learning Based NLP Project



## 🧠 Project Overview

This project aims to perform **coreference resolution in Turkish text** using a **statistical machine learning approach**.  

Coreference resolution is the task of identifying when different words or expressions in a text refer to the same entity.

Example:

> Ahmet kitabı aldı. O okumaya başladı.

Here, **“O” → “Ahmet”**

---

## 🎯 Objective

The main objectives of this project are:

- Detect coreference relations between words in Turkish sentences  
- Represent data in **CoNLL format**  
- Train a statistical machine learning model  
- Evaluate performance using standard NLP metrics  
- Generate BIO-tagged output and cluster-based entity chains  

---

## 📂 Dataset Format (CoNLL)

The dataset is formatted in **CoNLL style annotation**:



token_id   word   label



Example:



1   Ahmet   B-1
2   kitabı  I-1
3   raftan  O



### Label Description:
- **B-X** → Beginning of a coreference cluster  
- **I-X** → Inside a coreference cluster  
- **O** → No coreference relation  

---

## ⚙️ System Pipeline

The system consists of the following steps:

### 1. Data Loading
- Reads CoNLL formatted files
- Splits documents into token sequences

### 2. Feature Extraction
For each pair of words:

- Case-insensitive equality
- Capitalization similarity
- Substring similarity

### 3. Pairwise Classification
Each word pair is classified as:

- `1` → Coreference exists  
- `0` → No coreference  

### 4. Model Training
- Logistic Regression classifier
- Class imbalance handled using `class_weight="balanced"`

### 5. Clustering
- Union-Find algorithm
- Groups words into coreference chains

### 6. BIO Tagging
- Converts clusters into CoNLL BIO format

---

## 🤖 Machine Learning Model

- Algorithm: **Logistic Regression**
- Library: `scikit-learn`
- max_iter: 1000
- class_weight: balanced

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 📉 Results

### 🔹 Numerical Results



Accuracy: 0.94
Precision: 0.16
Recall: 0.18
F1-score: 0.17





### 📊 Confusion Matrix

📌 Insert image below after running project:

```md
![Confusion Matrix](results/confusion_matrix.png)


---

### 📈 Metrics Visualization

📌 Insert metric chart:

```md
![Evaluation Metrics](results/metrics.png)



## 📌 Interpretation of Results

Although the system achieves **high accuracy (~0.94)**, performance on the **coreference class (minority class)** is low.

### Why?

### ⚠️ 1. Class Imbalance

* Non-coreference (0): ~658 samples
* Coreference (1): ~22 samples

➡ Model biased toward majority class

---

### ⚠️ 2. Limited Feature Representation

Current features are:

* Surface-level matching
* No semantic embeddings
* No contextual understanding

---

### ⚠️ 3. Model Limitation

* Logistic Regression is linear
* Cannot capture deep semantic relations

---

## 📌 Output Format

The system generates CoNLL output:



results/final.conll



Example:

```

1   Ahmet   B-1
2   kitabı  I-1
6   O       I-1



---

## 📁 Project Structure

```

src/
│── main.py
│── utils.py
│── model.py

data/
│── train.conll
│── test.conll

results/
│── final.conll
│── confusion_matrix.png
│── metrics.png

```

---

## 🚀 How to Run

```bash
python src/main.py
```

---

## 📌 Limitations

* Dataset is highly imbalanced
* No semantic embeddings (e.g., BERT)
* Pairwise model ignores global context
* Rule-based clustering simplicity

---

## 🔮 Future Work

* 🔥 BERT / Sentence Transformer embeddings
* 🧠 CRF-based sequence labeling
* 📊 Data augmentation for minority class
* 🌐 Span-based neural coreference models
* 📈 Graph-based clustering methods

---

## 🧾 Conclusion

This project presents a **baseline statistical coreference resolution system for Turkish text**.

While the system is fully functional and produces valid CoNLL outputs, its performance is limited due to:

* Data imbalance
* Lack of semantic features
* Linear model constraints

Despite these limitations, the system successfully demonstrates an end-to-end NLP pipeline.

---

## 📷 Figures (Auto Generated)

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Evaluation Metrics

![Metrics](results/metrics.png)

