# ============================================
# AI RESUME SCREENING SYSTEM
# ============================================

# Step 1 - Import Libraries

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ============================================
# Step 2 - Load Dataset
# ============================================

df = pd.read_csv("resume_dataset.csv")

# ============================================
# Step 3 - Check Missing Values
# ============================================

print("Missing Values")
print(df.isnull().sum())

# Remove missing rows if any
df.dropna(inplace=True)

# ============================================
# Step 4 - Encode Category
# ============================================

encoder = LabelEncoder()

df["Category"] = encoder.fit_transform(df["Category"])

# ============================================
# Step 5 - Convert Resume Text into Numbers
# ============================================

tfidf = TfidfVectorizer(stop_words="english")

X = tfidf.fit_transform(df["Resume"])

y = df["Category"]

# ============================================
# Step 6 - Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================
# Step 7 - Train Model
# ============================================

model = MultinomialNB()

model.fit(X_train, y_train)

# ============================================
# Step 8 - Predict
# ============================================

prediction = model.predict(X_test)

# ============================================
# Step 9 - Accuracy
# ============================================

accuracy = accuracy_score(y_test, prediction)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ============================================
# Step 10 - Save Model
# ============================================

pickle.dump(model, open("resume_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("\nModel Saved Successfully!")

# ============================================
# Step 11 - Test with Your Own Resume
# ============================================

resume = input("\nPaste Resume Text:\n")

resume_data = tfidf.transform([resume])

result = model.predict(resume_data)

category = encoder.inverse_transform(result)

print("\nPredicted Category:", category[0])