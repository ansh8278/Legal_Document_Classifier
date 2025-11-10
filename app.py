from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from utils import extract_keywords, calculate_readability, highlight_keywords
import torch
import re

app = Flask(__name__)

# =========================================
# Configure Safe Device for M1
# =========================================
device = -1  # Force CPU for Apple Silicon stability
print("✅ Running on CPU mode (optimized for Mac M1/M2).")

# =========================================
# Load Lightweight NLP Models
# =========================================
summarizer = pipeline("summarization", model="t5-small", device=device)

classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",  # lightweight and stable for macOS
    device=device
)

# =========================================
# Helper Function: Document Classification
# =========================================
def classify_document(text):
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = cleaned[:1500]  # avoid overloading the model with long text

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
    
    # AI-based zero-shot classification
    classification = classifier(cleaned, candidate_labels=labels, multi_label=False)
    doc_type = classification["labels"][0]

    # 🔹 Rule-based refinement for more accuracy
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

    return doc_type  # ✅ only returning document type (no confidence)

# =========================================
# Flask Routes
# =========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    try:
        # 🔹 Summarization
        max_len = min(120, len(text) // 2)
        result = summarizer(text, max_length=max_len, min_length=20, do_sample=False)
        summary = result[0]['summary_text']

        # 🔹 Keyword Extraction and Readability
        keywords = extract_keywords(text)
        readability = calculate_readability(summary)
        highlighted = highlight_keywords(summary, keywords)

        # 🔹 Document Classification (without accuracy)
        doc_type = classify_document(text)

        # 🔹 JSON Response
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
# Run Flask App
# =========================================
if __name__ == '__main__':
    app.run(debug=True)
