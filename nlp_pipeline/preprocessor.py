"""Text preprocessing: tokenize, remove stopwords, lemmatize."""

import re
import string

import nltk

# Download once (with network timeout guard)
for resource in ["punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt_tab"
                       else f"corpora/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass  # network unavailable, skip — the user may need to run nltk.download() manually

stop_words = set(nltk.corpus.stopwords.words("english"))
try:
    lemmatizer = nltk.stem.WordNetLemmatizer()
except LookupError:
    lemmatizer = None  # wordnet data not available


def preprocess_text(text: str) -> str:
    """Clean and normalize raw text."""
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Lemmatize (skip if wordnet unavailable)
    if lemmatizer is not None:
        try:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except LookupError:
            pass  # wordnet data missing, keep raw tokens

    return " ".join(tokens)


def preprocess_news_df(df):
    """Add a 'cleaned_text' column by preprocessing title + content."""
    df = df.copy()
    df["cleaned_text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).apply(preprocess_text)
    return df
