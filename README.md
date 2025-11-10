LegalEase - Easy Starter (All-in-One)
====================================

This is a simplified, **easy-to-run** version of LegalEase. It contains:
- a single Flask app (`app.py`) that runs a demo summarizer + keywords + readability + doc-type (zero-shot)
- a small dataset (`data/sample_pairs.jsonl`) you can use to test immediately
- `requirements.txt` with minimal required packages
- instructions below so you can run it locally (no training required)

Notes:
- The app uses pre-trained transformer models and will download them the first time it runs (internet required).
- For faster startup on low-RAM machines, the app uses a distilled summarization model.

Quick start (Linux/macOS)
-------------------------
1. Unzip the project and cd into it:
   ```bash
   unzip LegalEase_Easy.zip -d .
   cd LegalEase_Easy
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser to `http://127.0.0.1:5000` and paste a legal paragraph (or click "Load sample").

Files included
--------------
- `app.py`            : single-file Flask app (inference + UI)
- `utils.py`          : helper functions (keywords, readability)
- `data/sample_pairs.jsonl` : 5 small legal-like examples
- `requirements.txt`  : pip packages
- `LegalEase_Easy.zip` : this zip (downloaded)

If you want a Colab notebook or Dockerfile, tell me and I'll add it.

