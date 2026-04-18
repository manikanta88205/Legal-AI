import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymupdf
from summarizer import generate_full_summary
from legal_analytics_engine import extract_full_analytics
from cleaner import clean_extracted_text
app = Flask(__name__)
# FIXED CORS CONFIG
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200
# ================================
# UPLOAD
# ================================
@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_file():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    raw_text = ""
    if file.filename.lower().endswith(".pdf"):
        pdf = pymupdf.open(filepath)
        for page in pdf:
            raw_text += page.get_text()
        pdf.close()
    else:
        return jsonify({"error": "Only PDF allowed"}), 400
    extracted_text = clean_extracted_text(raw_text)
    return jsonify({
        "fileName": file.filename,
        "rawText": raw_text,
        "extractedText": extracted_text
    }), 200
# ================================
# SUMMARY
# ================================
@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize_text():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    summary_data = generate_full_summary(data["text"])
    return jsonify(summary_data), 200
# ================================
# ENTITY EXTRACTION
# ================================
@app.route("/extract_entities", methods=["POST", "OPTIONS"])
def extract_entities():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    analytics = extract_full_analytics(data["text"])
    return jsonify(analytics), 200
# ================================
# FULL ANALYSIS
# ================================
@app.route("/full_analysis", methods=["POST", "OPTIONS"])
def full_analysis():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    analytics = extract_full_analytics(data["text"])
    summary_data = generate_full_summary(data["text"])
    return jsonify({
        "analytics": analytics,
        "summary": summary_data
    }), 200
# ================================
# CHATBOT
# ================================
@app.route("/chatbot", methods=["POST", "OPTIONS"])
def chatbot():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    data = request.json
    if "question" not in data or "context" not in data:
        return jsonify({"error": "Question and context required"}), 400
    
    question = data["question"]
    context = data["context"]
    
    if not context:
        return jsonify({"answer": "Please upload a document first to ask questions."}), 200
    
    # Simple keyword-based Q&A system
    answer = generate_answer(question, context)
    
    return jsonify({"answer": answer}), 200

def generate_answer(question: str, context: str) -> str:
    """Generate answer based on question and context."""
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Extract relevant sentences from context
    sentences = [s.strip() for s in context.split('.') if s.strip()]
    
    # Find sentences containing keywords from the question
    question_words = [w for w in question_lower.split() if len(w) > 3]
    
    relevant_sentences = []
    for sentence in sentences:
        match_count = sum(1 for word in question_words if word in sentence)
        if match_count > 0:
            relevant_sentences.append((sentence, match_count))
    
    # Sort by relevance and get top matches
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_sentences:
        answer = ". ".join([s[0] for s in relevant_sentences[:3]]) + "."
        return answer if answer else "I couldn't find relevant information in the document."
    
    return "I couldn't find relevant information for your question. Please try asking something related to the document."

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
