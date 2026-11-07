"""
Lab 1: Introduction to Deep Learning Concepts
Practical: Traditional Machine Learning with Bag of Words.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Sample training data
texts = [
    "I love this product",
    "This is an excellent product",
    "I am very happy",
    "The product is amazing",
    "I hate this product",
    "This is a terrible product",
    "I am very disappointed",
    "The product is bad"
]

labels = [
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative",
    "Negative"
]

# 2. Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nBag of Words feature matrix:")
print(X.toarray())

# 3. Train a traditional ML classifier
model = LogisticRegression()
model.fit(X, labels)

# 4. Test the trained model
test_sentences = [
    "I love this amazing product",
    "This product is terrible"
]

test_features = vectorizer.transform(test_sentences)
predictions = model.predict(test_features)

print("\nSentiment Predictions:")

for sentence, prediction in zip(test_sentences, predictions):
    print(f"{sentence} -> {prediction}")

print("\nTraditional ML practical completed successfully!")
