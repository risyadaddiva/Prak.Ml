# %% Load Library
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %% Load IMDb dataset
dataset = load_dataset("imdb")

# Convert to DataFrame
df = pd.DataFrame(dataset["train"])  # Hanya ambil data training IMDb
print(df.head())

# %% Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)

# %% Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %% Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# %% Predict
y_pred = model.predict(X_test_vec)

# %% Evaluation
print(classification_report(y_test, y_pred))

# %% Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# %%
