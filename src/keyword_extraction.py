from sklearn.feature_extraction.text import CountVectorizer


def extract_top_ngrams(corpus, ngram_range=(1, 2), top_k=20) -> list:
    """
    Extracts top n-grams (unigrams, bigrams) from a list of cleaned text.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    freq_dict = dict(zip(vocab, counts))
    sorted_ngrams = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_ngrams[:top_k]
