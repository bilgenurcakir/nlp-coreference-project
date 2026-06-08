import os
import numpy as np
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score



# 1. CONLL READER

def conll_oku(path):
    docs, doc = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if doc:
                    docs.append(doc)
                    doc = []
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                doc.append(parts)

    if doc:
        docs.append(doc)

    return docs


# 2. FEATURES

def features(w1, w2):
    return [
        int(w1.lower() == w2.lower()),
        int(w1[0].isupper() and w2[0].isupper()),
        int(w1.lower() in w2.lower() or w2.lower() in w1.lower())
    ]


def create_pairs(docs):
    X, y = [], []

    for doc in docs:
        for i, j in itertools.combinations(range(len(doc)), 2):
            w1, c1 = doc[i][1], doc[i][2]
            w2, c2 = doc[j][1], doc[j][2]

            label = 1 if (c1 != "_" and c1 == c2) else 0

            X.append(features(w1, w2))
            y.append(label)

    return np.array(X), np.array(y)


# 3. MODEL

def train_model(X, y):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model


# 4. UNION FIND

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# 5. CLUSTERING

def build_clusters(docs, model):
    all_clusters = []

    for doc in docs:
        uf = UnionFind()

        for i, j in itertools.combinations(range(len(doc)), 2):

            w1, w2 = doc[i][1], doc[j][1]
            x = np.array(features(w1, w2)).reshape(1, -1)

            prob = model.predict_proba(x)[0][1]

            if prob > 0.5:
                uf.union(i, j)

        clusters = defaultdict(list)
        for i in range(len(doc)):
            clusters[uf.find(i)].append(i)

        all_clusters.append(clusters)

    return all_clusters


# 6. BIO + CoNLL WRITER (FIXED)

def write_conll(docs, clusters, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:

        for doc_id, doc in enumerate(docs):

            cluster_map = {}
            cluster_id = 1

            for root, members in clusters[doc_id].items():
                for idx, m in enumerate(members):
                    if idx == 0:
                        cluster_map[m] = ("B", cluster_id)
                    else:
                        cluster_map[m] = ("I", cluster_id)
                cluster_id += 1

            for i, token in enumerate(doc):
                word = token[1]

                if i not in cluster_map:
                    label = "O"
                else:
                    bi, cid = cluster_map[i]
                    label = f"{bi}-{cid}"

                f.write(f"{i + 1}\t{word}\t{label}\n")

            f.write("\n")


# 7. METRICS + PLOTS

def evaluate_and_plot(model, X, y):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print("\nRESULTS")
    print("====================")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print(classification_report(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)

    # CONFUSION MATRIX PLOT
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # METRICS BAR PLOT
    plt.figure()
    plt.bar(["Accuracy", "Precision", "Recall", "F1"], [acc, prec, rec, f1])
    plt.title("Evaluation Metrics")
    plt.savefig("results/metrics.png")
    plt.close()


# 8. MAIN

def main():
    print(">>> PROGRAM STARTED <<<")

    TRAIN_PATH = "data/train.conll"
    TEST_PATH = "data/test.conll"

    print("Loading data...")
    train_docs = conll_oku(TRAIN_PATH)
    test_docs = conll_oku(TEST_PATH)

    print("Feature extraction...")
    X_train, y_train = create_pairs(train_docs)
    X_test, y_test = create_pairs(test_docs)

    print("Train samples:", len(X_train))
    print("Test samples :", len(X_test))

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluation...")
    evaluate_and_plot(model, X_test, y_test)

    print("Clustering...")
    clusters = build_clusters(test_docs, model)

    print("Writing CoNLL...")
    write_conll(test_docs, clusters, "results/final.conll")

    print("DONE -> results/final.conll")


if __name__ == "__main__":
    main()