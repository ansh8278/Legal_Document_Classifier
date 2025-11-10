from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from utils import extract_keywords, calculate_readability, highlight_keywords
import torch
import re

app = Flask(__name__)

# =========================================
# Configure Device (Auto-detect MPS for Mac)
# =========================================
if torch.backends.mps.is_available():
    device = 0  # Use Apple Metal GPU
    print("✅ Running on MPS (Apple Silicon GPU).")
else:
    device = -1  # CPU fallback
    print("✅ Running on CPU mode (safe for Mac M1/M2).")

# =========================================
# Load NLP Pipelines
# =========================================
print("🔄 Loading NLP models... please wait...")

# Summarizer
summarizer = pipeline("summarization", model="t5-small", device=device)

# Classifier
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
    device=device
)

# Question-Answering model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=device
)

print("✅ Models loaded successfully!")

# =========================================
# Helper: Document Classification
# =========================================
def classify_document(text):
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = cleaned[:1500]

    labels = [
        "legal agreement",
        "contract document",
        "official notice",
        "privacy policy",
        "court judgment",
        "employment offer letter",
        "terms and conditions",
        "legal affidavit",
        "service level agreement",
        "memorandum of understanding"
    ]

    classification = classifier(cleaned, candidate_labels=labels, multi_label=False)
    doc_type = classification["labels"][0]

    # Rule-based refinement
    lower_text = text.lower()
    if any(k in lower_text for k in ["court", "judge", "tribunal", "case number"]):
        doc_type = "court judgment"
    elif any(k in lower_text for k in ["privacy", "policy", "data protection"]):
        doc_type = "privacy policy"
    elif any(k in lower_text for k in ["agreement", "contract", "party", "obligation"]):
        doc_type = "legal agreement"
    elif any(k in lower_text for k in ["notice", "hereby", "serve notice"]):
        doc_type = "official notice"
    elif any(k in lower_text for k in ["employment", "employee", "offer letter"]):
        doc_type = "employment offer letter"
    elif any(k in lower_text for k in ["terms", "conditions", "usage", "agreement of service"]):
        doc_type = "terms and conditions"

    return doc_type


# =========================================
# ROUTE 1: Home Page
# =========================================
@app.route('/')
def index():
    return render_template('index.html')


# =========================================
# ROUTE 2: Simplify Document
# =========================================
@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    try:
        # Summarize document
        max_len = min(120, len(text) // 2)
        result = summarizer(text, max_length=max_len, min_length=20, do_sample=False)
        summary = result[0]['summary_text']

        # Extract keywords, readability, and highlight
        keywords = extract_keywords(text)
        readability = calculate_readability(summary)
        highlighted = highlight_keywords(summary, keywords)

        # Document classification
        doc_type = classify_document(text)

        return jsonify({
            "summary": summary,
            "highlighted": highlighted,
            "keywords": keywords,
            "readability": readability,
            "doc_type": doc_type
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# =========================================
# ROUTE 3: Question Answering Feature
# =========================================
@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer user’s natural-language questions from document text."""
    data = request.get_json()
    text = data.get('text', '').strip()
    question = data.get('question', '').strip()

    if not text or not question:
        return jsonify({"error": "Please provide both document text and a question."}), 400

    try:
        # Run the QA model
        answer = qa_pipeline(question=question, context=text)

        return jsonify({
            "question": question,
            "answer": answer.get("answer", "No answer found."),
            "confidence": round(answer.get("score", 0.0) * 100, 2)
        })

    except Exception as e:
        print("❌ QA Error:", e)
        return jsonify({"error": f"Unable to process question: {str(e)}"}), 500


# =========================================
# Run Flask App
# =========================================
if __name__ == '__main__':
    app.run(debug=True)
