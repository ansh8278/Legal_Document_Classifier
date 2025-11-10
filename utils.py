from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_kincaid_grade
import re

def extract_keywords(text, num_keywords=5):
    """Extracts top keywords using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        scores = tfidf_matrix.toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[-num_keywords:][::-1]
        return [feature_names[i] for i in top_indices]
    except Exception:
        return []

def calculate_readability(text):
    """Computes Flesch-Kincaid grade level."""
    try:
        grade = flesch_kincaid_grade(text)
        if grade < 6:
            level = "Very Easy"
        elif grade < 9:
            level = "Easy"
        elif grade < 12:
            level = "Medium"
        else:
            level = "Difficult"
        return f"{level} (Grade {grade:.1f})"
    except Exception:
        return "N/A"

def highlight_keywords(text, keywords):
    """Highlights keywords in summary text."""
    for kw in keywords:
        pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
        text = pattern.sub(r'<mark>\1</mark>', text)
    return text
