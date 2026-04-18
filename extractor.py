import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# ===========================================================
# UTILITY FUNCTIONS
# ===========================================================

def clean_spaces(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_block(text, keywords):
    """Extract paragraphs around a keyword block."""
    pattern = r"(?s)(?:{})[^\n]*\n(.+?)(?=\n[A-Z][A-Za-z ]{{2,}}:|\n[A-Z]|\Z)".format("|".join(keywords))
    match = re.search(pattern, text, re.I)
    return match.group(1).strip() if match else ""


# ===========================================================
# 1. COURT, CASE TITLE & PARTIES
# ===========================================================

def extract_case_metadata(text):
    metadata = {
        "court": "Not found",
        "case_title": "",
        "case_number": "",
        "judgment_date": "",
        "bench": [],
        "parties": {
            "appellant": "",
            "respondent": ""
        },
    }

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Court
    for line in lines:
        u = line.upper()
        if "SUPREME COURT" in u:
            metadata["court"] = "Supreme Court of India"
            break
        if "HIGH COURT" in u:
            metadata["court"] = line
            break

    # Case number
    case_no = re.findall(r"[A-Z.() ]*\d+\/\d{4}", text)
    metadata["case_number"] = case_no[0] if case_no else ""

    # Date
    date = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", text)
    metadata["judgment_date"] = date[0] if date else ""

    # Judges
    judge_pattern = r"(JUSTICE\s+[A-Z][A-Za-z. ]+)"
    judges = re.findall(judge_pattern, text, re.I)
    metadata["bench"] = list(set([j.strip() for j in judges]))

    # Parties via VERSUS
    for i, line in enumerate(lines):
        if re.fullmatch(r"(versus|vs|v\.?)", line, re.I):
            if i > 0 and i < len(lines) - 1:
                metadata["parties"]["appellant"] = lines[i-1]
                metadata["parties"]["respondent"] = lines[i+1]
            break

    metadata["case_title"] = f"{metadata['parties']['appellant']} vs {metadata['parties']['respondent']}"
    return metadata


# ===========================================================
# 2. CITATIONS, SECTIONS, ACTS
# ===========================================================

def extract_statutes(text):
    act_pattern = r"[A-Za-z ]+Act,?\s*\d{4}"
    return list(set(re.findall(act_pattern, text)))


def extract_sections(text):
    pattern = r"(Section\s+\d+[A-Za-z]*\s*(?:of\s+[A-Za-z ]+)?(?:Act)?)"
    return list(set(re.findall(pattern, text)))


def extract_citations(text):
    citation_pattern = (
        r"(\d{4}\s*SCC\s*Online\s*\w+\s*\d+|" 
        r"\(\d{4}\)\s*\d+\s*SCC\s*\d+|"  
        r"AIR\s*\d{4}\s*\w+\s*\d+|"  
        r"MANU\/\w+\/\d+\/\d+|"  
        r"\d+\s*Cri\s*LJ\s*\d+)"
    )
    return list(set(re.findall(citation_pattern, text)))


# ===========================================================
# 3. ISSUES, FACTS, ARGUMENTS, REASONING, DECISION
# ===========================================================

def extract_issues(text):
    return extract_block(text, ["Issues", "Points for determination", "Questions"])


def extract_facts(text):
    return extract_block(text, ["Background", "Facts", "Incident", "Case of"])


def extract_arguments(text):
    return {
        "appellant": extract_block(text, ["Appellant argued", "Contention of appellant", "Learned counsel for appellant"]),
        "respondent": extract_block(text, ["Respondent argued", "Contention of respondent", "Learned counsel for respondent"]),
    }


def extract_reasoning(text):
    return extract_block(text, ["Analysis", "Reasoning", "Discussion", "Findings"])


def extract_final_order(text):
    order_pattern = r"(appeal (?:allowed|dismissed|disposed|partly allowed)|compensation.+?awarded)"
    match = re.search(order_pattern, text, re.I)
    return match.group(1) if match else ""


# ===========================================================
# 4. COMPENSATION TABLE EXTRACTION
# ===========================================================

def extract_compensation(text):
    amounts = re.findall(r"Rs\.?\s*[\d,]+", text)
    return {
        "amounts": amounts[:10],   # limit to top 10
        "highest_amount": max(amounts, default="")
    }


# ===========================================================
# 5. SUMMARY ENGINE (bullet-point)
# ===========================================================

def generate_summary(text, bullets=5):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return text

    freq = {}
    words = word_tokenize(text.lower())

    for w in words:
        if w.isalpha() and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    max_freq = max(freq.values())
    freq = {k: v / max_freq for k, v in freq.items()}

    scores = {}
    for s in sentences:
        for w in word_tokenize(s.lower()):
            if w in freq:
                scores[s] = scores.get(s, 0) + freq[w]

    ranked = sorted(scores, key=scores.get, reverse=True)[:bullets]
    ranked = sorted(ranked, key=lambda s: sentences.index(s))

    return "\n\n• " + "\n• ".join(ranked)


# ===========================================================
# 6. MAIN PIPELINE
# ===========================================================

def extract_full_analytics(text):
    text = clean_spaces(text)

    return {
        "metadata": extract_case_metadata(text),
        "sections": extract_sections(text),
        "acts": extract_statutes(text),
        "citations": extract_citations(text),
        "facts": extract_facts(text),
        "issues": extract_issues(text),
        "arguments": extract_arguments(text),
        "reasoning": extract_reasoning(text),
        "final_order": extract_final_order(text),
        "compensation": extract_compensation(text),
        "summary": generate_summary(text),
    }