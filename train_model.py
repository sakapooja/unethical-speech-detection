import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

# 1. Load Dataset (Wikipedia/Twitter Research Data)
url = 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv'
print("Downloading dataset...")
df = pd.read_csv(url)

# Labeling: 0 & 1 -> Harmful (1), 2 -> Safe (0)
df['label'] = df['class'].apply(lambda x: 1 if x < 2 else 0)

# 2. Balancing the data
harmful_df = df[df['label'] == 1]
safe_df = df[df['label'] == 0]
sample_size = min(len(harmful_df), len(safe_df), 10000)

df_balanced = pd.concat([
    harmful_df.sample(sample_size, random_state=42),
    safe_df.sample(sample_size, random_state=42)
])

# 3. Vectorization (Bi-grams for context)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df_balanced['tweet'])
y = df_balanced['label']

# 4. Training SVM (Balanced weights to increase sensitivity)
model = LinearSVC(C=1.0, class_weight='balanced', random_state=42)
model.fit(X, y)

# 5. Save the 'Brain'
with open('svm_model_final.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_final.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Training complete! Accuracy: {model.score(X, y)*100:.2f}%")
