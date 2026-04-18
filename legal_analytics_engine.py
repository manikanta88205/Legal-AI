import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


# ===========================================================
# CLEAN HELPERS
# ===========================================================

def clean_spaces(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def smart_block(text, keywords):
    """
    Extracts a short 3–8 line block only under the keyword,
    without capturing huge paragraphs.
    """
    pattern = r"(?si)(?:^|\n)({})(?:[:. ])\s*(.*?)(?=\n[A-Z][A-Za-z ]+[:.]|\n\d+\.)".format(
        "|".join(keywords)
    )
    match = re.search(pattern, text)
    if not match:
        return ""

    block = match.group(2).strip()
    lines = block.split("\n")

    # Limit to top 6 relevant lines
    trimmed = "\n".join(lines[:6])
    return trimmed.strip()


# ===========================================================
# CASE METADATA
# ===========================================================

def extract_case_metadata(text):
    metadata = {
        "court": "",
        "case_title": "",
        "case_number": "",
        "judgment_date": "",
        "bench": [],
        "parties": {"appellant": "", "respondent": ""}
    }

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Court name
    for l in lines:
        if "SUPREME COURT" in l.upper():
            metadata["court"] = "Supreme Court of India"
            break
        if "HIGH COURT" in l.upper():
            metadata["court"] = l
            break

    # Case number
    case_no = re.findall(r"[A-Za-z ()]*\d+\/\d{4}", text)
    metadata["case_number"] = case_no[0] if case_no else ""

    # Judgment date
    date = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", text)
    metadata["judgment_date"] = date[0] if date else ""

    # Bench / judges
    judges = re.findall(r"JUSTICE\s+[A-Z][A-Za-z .]+", text, flags=re.I)
    metadata["bench"] = list(set(judges))

    # Parties (Appellant / Respondent)
    party_app = re.findall(r"(.*?)\s*[.…]+\s*Appellant", text, re.I)
    party_res = re.findall(r"(.*?)\s*[.…]+\s*Respondent", text, re.I)

    if party_app:
        metadata["parties"]["appellant"] = party_app[0].strip()

    if party_res:
        metadata["parties"]["respondent"] = party_res[0].strip()

    # Fallback using V
    for i, l in enumerate(lines):
        if l.lower() in ["versus", "vs", "v."]:
            metadata["parties"]["appellant"] = metadata["parties"]["appellant"] or lines[i-1]
            metadata["parties"]["respondent"] = metadata["parties"]["respondent"] or lines[i+1]
            break

    # Title
    if metadata["parties"]["appellant"] and metadata["parties"]["respondent"]:
        metadata["case_title"] = (
            metadata["parties"]["appellant"] + " vs " + metadata["parties"]["respondent"]
        )

    return metadata


# ===========================================================
# LEGAL EXTRACTORS
# ===========================================================

def extract_sections(text):
    return list(set(re.findall(r"Section\s+\d+[A-Za-z]*", text)))


def extract_statutes(text):
    acts = re.findall(r"[A-Za-z ]+Act,?\s*\d{4}", text)
    return list(set([a.strip() for a in acts]))


def extract_citations(text):
    pattern = (
        r"(?:\(\d{4}\)\s*\d+\s*SCC\s*\d+|"
        r"\d{4}\s*SCC\s*Online\s*[A-Za-z]+\s*\d+|"
        r"AIR\s*\d{4}\s*[A-Za-z]+\s*\d+|"
        r"MANU\/[A-Z]+\/\d+\/\d+)"
    )
    return list(set(re.findall(pattern, text)))


# ===========================================================
# FACTS / ISSUES / REASONING
# ===========================================================

def extract_facts(text):
    block = smart_block(text, ["FACTS", "BACKGROUND", "FACTUAL MATRIX"])
    return block or "Not extracted."


def extract_issues(text):
    block = smart_block(text, ["ISSUES", "QUESTIONS", "POINTS FOR DETERMINATION"])
    return block or "Not specified."


def extract_arguments(text):
    return {
        "appellant": smart_block(text, ["APPELLANT", "ARGUMENTS OF APPELLANT"]),
        "respondent": smart_block(text, ["RESPONDENT", "ARGUMENTS OF RESPONDENT"])
    }


def extract_reasoning(text):
    block = smart_block(text, ["REASONING", "DISCUSSION", "ANALYSIS", "FINDINGS"])
    return block or "Not extracted."


def extract_final_order(text):
    pattern = (
        r"(appeal.*?allowed|appeal.*?dismissed|compensation.*?awarded|"
        r"petition.*?allowed|petition.*?dismissed)"
    )
    m = re.search(pattern, text, re.I)
    return m.group(1) if m else "Not detected."


# ===========================================================
# COMPENSATION
# ===========================================================

def extract_compensation(text):
    """
    Extract compensation ONLY if the judgment contains
    keywords like 'compensation', 'award'.
    """
    if not re.search(r"compensation|award", text, re.I):
        return {"amounts": [], "highest_amount": ""}

    amounts = re.findall(r"Rs\.?\s*[\d,]+", text)
    values = [int(re.sub(r"[^\d]", "", a)) for a in amounts]

    return {
        "amounts": amounts[:8],
        "highest_amount": max(values) if values else ""
    }


# ===========================================================
# SUMMARY ENGINE (SHORT)
# ===========================================================

def generate_short_summary(text):
    sentences = sent_tokenize(text)
    top = sentences[:6]
    return "\n".join("• " + s for s in top)


# ===========================================================
# MASTER PIPELINE
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

        "compensation": extract_compensation(text)

        
    }
