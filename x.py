import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import re
import string

# 讀取數據
df = pd.read_csv("IMDB Dataset.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df['review'] = df['review'].apply(clean_text)

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

# TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 訓練模型
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)

gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_train_tfidf, y_train)

# 預測與評估
rf_preds = rf.predict(X_test_tfidf)
gb_preds = gb.predict(X_test_tfidf)

rf_acc = accuracy_score(y_test, rf_preds)
gb_acc = accuracy_score(y_test, gb_preds)

print(f"RandomForest Accuracy: {rf_acc:.4f}")
print(f"GradientBoosting Accuracy: {gb_acc:.4f}")
