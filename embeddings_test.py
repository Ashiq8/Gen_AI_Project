from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'This is the first document.',
    'This document is thee second document',
    'And this is the third one',
    'Carun is the good boy of thee class'
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)

print(X.toarray())
print(vectorizer.get_feature_names_out())