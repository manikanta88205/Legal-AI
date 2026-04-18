import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = set(stopwords.words("english"))

# ===========================================================
# 1. CLEANING PIPELINE
# ===========================================================

def clean_for_summary(text: str) -> str:
    """Remove noise but preserve structure (newlines, paragraphs)."""

    # Remove citations (SCC, SCC Online, AIR, MANU, Cri LJ, etc.)
    citation_patterns = [
        r"\(\d{4}\)\s*\d+\s*SCC.*?\d+",
        r"\d{4}\s*SCC\s*Online\s*[A-Za-z]+\s*\d+",
        r"AIR\s*\d{4}\s*[A-Za-z]+\s*\d+",
        r"MANU\/\w+\/\d+\/\d+",
        r"\d+\s*Cri\s*LJ\s*\d+",
    ]
    for p in citation_patterns:
        text = re.sub(p, " ", text, flags=re.I)

    # Remove page numbers, serial numbers
    text = re.sub(r"\n?\s*\b\d+\.\s", "\n", text)

    # Remove repeated dots / lines
    text = re.sub(r"[·•…]+", "", text)

    # Remove multiple spaces but PRESERVE newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ===========================================================
# 2. SENTENCE SCORING
# ===========================================================

def sentence_scores(text: str):
    sentences = sent_tokenize(text)
    if not sentences:
        return {}, []

    words = word_tokenize(text.lower())

    freq = {}
    for w in words:
        if w.isalpha() and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        return {}, sentences

    max_freq = max(freq.values())
    freq = {k: v / max_freq for k, v in freq.items()}

    scores = {}
    for s in sentences:
        for w in word_tokenize(s.lower()):
            if w in freq:
                scores[s] = scores.get(s) + freq[w] if s in scores else freq[w]

    return scores, sentences


# ===========================================================
# 3. BULLET SUMMARY GENERATOR
# ===========================================================

def generate_bullet_summary(text: str, count: int = 5) -> str:
    cleaned = clean_for_summary(text)
    scores, sentences = sentence_scores(cleaned)

    if not scores:
        return cleaned

    ranked = sorted(scores, key=scores.get, reverse=True)[:count]
    ranked = sorted(ranked, key=lambda s: sentences.index(s))

    bullets = "\n\n• " + "\n• ".join(ranked)
    return bullets


# ===========================================================
# 4. PARAGRAPH SUMMARY
# ===========================================================

def generate_paragraph_summary(text: str, count: int = 5) -> str:
    cleaned = clean_for_summary(text)
    scores, sentences = sentence_scores(cleaned)

    if not scores:
        return cleaned

    ranked = sorted(scores, key=scores.get, reverse=True)[:count]
    ranked = sorted(ranked, key=lambda s: sentences.index(s))

    return " ".join(ranked)


# ===========================================================
# 5. ISSUE-BASED SUMMARY
# ===========================================================

def extract_issues_for_summary(text: str):
    pattern = r"(?s)(Issues?|Questions?|Points for determination).*?:?(.*?)(?=\n[A-Z]|$)"
    match = re.search(pattern, text, re.I)
    if match:
        return match.group(2).strip()
    return ""


def generate_issue_summary(text: str):
    issues = extract_issues_for_summary(text)
    if not issues:
        return "No specific issues mentioned."
    points = re.split(r"\n|;", issues)
    points = [p.strip() for p in points if len(p.strip()) > 3]
    return "\n\n• " + "\n• ".join(points)


# ===========================================================
# 6. FINAL PRO SUMMARY PIPELINE
# ===========================================================

def generate_full_summary(text: str):
    """Returns ALL types of summaries."""
    
    return {
        "bullet_summary": generate_bullet_summary(text),
        "paragraph_summary": generate_paragraph_summary(text),
        "issues_summary": generate_issue_summary(text)
    }
