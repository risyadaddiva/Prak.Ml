# %%
import pandas as pd

# %%
from sklearn.model_selection import train_test_split
# %%
from sklearn.tree import DecisionTreeClassifier
# %%
from sklearn.metrics import classification_report, confusion_matrix
# %%
import seaborn as sns
# %%
import matplotlib.pyplot as plt

# %%
# Load dataset
df = pd.read_csv("citrus.csv")
df = df[df['name'].isin(['orange', 'grapefruit'])]
df['label'] = df['name'].map({'orange': 0, 'grapefruit': 1})

# %%
# Filter hanya jeruk dan anggur
df = df[df['name'].isin(['orange', 'grapefruit'])]
print(df['name'].value_counts())

# %%
# Mapping label
df['label'] = df['name'].map({'orange': 0, 'grapefruit': 1})
df = df.dropna(subset=['label'])  # Drop rows with NaN in the label column
print(df['label'].value_counts())
print(df.head())


# %%
# Fitur dan target
X = df.drop(['name', 'label'], axis=1)
y = df['label']
print(X.head())
print(y.head())

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# %%
# Model decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# %%
# Prediksi dan evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['orange', 'grapefruit'], labels=[0, 1]))

# %%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['orange', 'grapefruit'], yticklabels=['orange', 'grape'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %%
