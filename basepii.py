"""
PII Scanner — Heuristic detection + NLI-based risk scoring
Dependencies:
    pip install spacy transformers torch
    python -m spacy download en_core_web_sm
"""

import re
import spacy
from transformers import pipeline
from functools import lru_cache

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")

# Model preference order: fastest/smallest → most accurate
# All are public, no HF token needed
NLI_MODELS = [
    "typeform/distilbert-base-uncased-mnli",   # ~250MB, very fast, no auth issues
    "cross-encoder/nli-deberta-v3-small",       # ~180MB, best accuracy
    "facebook/bart-large-mnli",                 # ~1.6GB, fallback
]

def load_nli(models=NLI_MODELS):
    for model_id in models:
        try:
            print(f"Loading NLI model: {model_id}")
            p = pipeline("zero-shot-classification", model=model_id, device=-1)
            print(f"✓ Loaded {model_id}")
            return p
        except Exception as e:
            print(f"✗ Failed ({model_id}): {e}")
    raise RuntimeError("All NLI models failed to load. Check your internet connection.")

nli = load_nli()

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "SSN":     r"\b\d{3}-\d{2}-\d{4}\b",
    "EMAIL":   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "PHONE":   r"\b(\+91[-\s]?)?[6-9]\d{9}\b",
}

# ---------------------------------------------------------------------------
# NLI intent labels
# The model picks the one that best describes the sentence context.
# "genuine exposure" → high risk
# "example / documentation" → low risk
# "technical reference" → medium risk (e.g. regex tutorials)
# ---------------------------------------------------------------------------

CANDIDATE_LABELS = [
    "sharing or leaking real personal information",   # → high risk
    "educational example or documentation",           # → low risk
    "technical reference or format description",      # → medium risk
]

LABEL_RISK_BASE = {
    "sharing or leaking real personal information":  0.85,
    "technical reference or format description":     0.55,
    "educational example or documentation":          0.20,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sentence(doc, span_start: int, span_end: int) -> str:
    for sent in doc.sents:
        if span_start >= sent.start_char and span_end <= sent.end_char:
            return sent.text
    return ""


def get_nearby_ner(doc, span_start: int, span_end: int) -> list[str]:
    """Return NER label types of entities in the same sentence."""
    labels = []
    for ent in doc.ents:
        if ent.start_char >= span_start - 200 and ent.end_char <= span_end + 200:
            labels.append(ent.label_)
    return labels


@lru_cache(maxsize=512)
def nli_score(sentence: str) -> tuple[str, float]:
    """
    Cache results so repeated identical sentences don't re-run inference.
    Returns (top_label, confidence).
    """
    result = nli(sentence, candidate_labels=CANDIDATE_LABELS)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return top_label, top_score


def compute_risk(sentence: str, nearby_ner: list[str]) -> float:
    """
    Primary signal: NLI intent classification.
    Secondary signal: spaCy NER context (PERSON / ORG nearby → risk up).
    """
    if not sentence.strip():
        return 0.6  # no context → treat as suspicious by default

    top_label, confidence = nli_score(sentence)
    base = LABEL_RISK_BASE[top_label]

    # Scale by model confidence — uncertain predictions stay closer to 0.5
    risk = base * confidence + 0.5 * (1 - confidence)

    # NER boost: real person or org name near the PII → slightly higher risk
    if "PERSON" in nearby_ner or "ORG" in nearby_ner:
        risk = min(risk + 0.08, 1.0)

    return round(risk, 3)


def severity(score: float) -> str:
    if score >= 0.75:
        return "CONFIRMED_LEAK"
    elif score >= 0.45:
        return "SUSPICIOUS"
    else:
        return "LOW_RISK"

# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def hybrid_scan(text: str) -> list[dict]:
    doc = nlp(text)
    findings = []

    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            span_start, span_end = match.start(), match.end()
            sentence    = get_sentence(doc, span_start, span_end)
            nearby_ner  = get_nearby_ner(doc, span_start, span_end)
            risk_score  = compute_risk(sentence, tuple(nearby_ner))

            findings.append({
                "type":       pii_type,
                "value":      match.group(),
                "sentence":   sentence,
                "risk_score": risk_score,
                "severity":   severity(risk_score),
                "nli_label":  nli_score(sentence)[0] if sentence.strip() else "no context",
                "ner_nearby": nearby_ner,
            })

    return findings

# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
    "My Aadhaar is 3456 7890 1234 – please don't save or forward it.",
    "Aadhaar number 876543210987 should be kept confidential.",
    "Kindly use 5678 9012 3456 for KYC but delete after use.",
    "Aadhaar 4321-0987-6543 was mistakenly shared in email.",
    "Don't ever post full Aadhaar like 9999 0000 1111 anywhere.",
    "PAN CARD ABCDE6789F – this is sensitive, handle carefully.",
    "My PAN is FGHIJ1234K, submitting only for this transaction.",
    "PAN number XYZAB5678P – please mask before sharing screenshot.",
    "Permanent Account Number : BCDER9012Q (do not circulate)",
    "PAN ABCPS9999R was accidentally visible in the zoom background.",
    "Reach me at 93999 12345 – this is my personal number.",
    "Mobile: +91 81456 78901 – only for today's discussion.",
    "Contact 77990-11223 – kindly delete chat after confirmation.",
    "Her number is 6281 234 5678, added by mistake to group.",
    "WhatsApp: 9876540321 – don't share with anyone else.",
    "Email me documents on priya.kumar2023@outlook.com",
    "Official mail: finance.team@startup.in – confidential only.",
    "Please use temp.testing2025@gmail.com for this demo.",
    "My recovery email secondary.user@yahoo.in got exposed.",
    "Passport number M12345678 – for hotel booking reference only.",
    "Indian Passport No. N98765432 expires 15/08/2031.",
    "Passport P45678901 – never upload unmasked copy online.",
    "DL number TS05 20230098765 – driving license details.",
    "Driving Licence No. MH1420220012345 valid till 2029.",
    "Bank a/c 9200200123456789 IFSC SBIN0001234",
    "Account Number 50100098765432 – transfer only ₹1 to verify.",
    "Beneficiary A/c 101234567890 – IFSC ICIC0000101 please confirm.",
    "Credit card 4346 7890 1234 5678 expiry 09/27 CVV 456",
    "Card ending **** **** **** 9012 – valid thru 12/26.",
    "Debit card 5243 6801 2345 6789 – testing payment gateway.",
    "Voter ID LBJ2345678 – EPIC number for address proof.",
    "EPIC No. ABC9876543 – please don't post publicly.",
    "Aadhaar 6789 0123 4567 – I know I shouldn't share but urgent.",
    "My full Aadhaar is 1111 2222 3333 – obviously dummy but still.",
    "Phone 9XX XXX X210 – last digits only, never post full.",
    "Aadhaar XXXX XXXX 9876 – is this safe to show?",
    "PAN ****P1234K – masked version, but real is WXYZA5678P",
    "Card 6011********1234 – only last 4 for verification.",
    "SSN 987-65-4321 used as dummy data – do NOT use real ones.",
    "Her Aadhaar 5432 1098 7654 appeared in public gist by mistake.",
    "Don't share my number +919000112233 under any circumstances.",
    "Email confidential@personal.in was leaked in old forum post.",
    "Passport J87654321 – photo copy has number visible, careful.",
    "Bank account 624301234567 – IFSC CORP0000624 transfer pending.",
    "Aadhaar card 8901 2345 6789 shared in family WhatsApp group.",
    "Please redact PAN BDFGH4321L before sending to client.",
    "Mobile no. 83290 45678 – OTP will come here, never forward.",
    "Voter ID ends with 3456 full is KLM9876543 – avoid posting.",
    "Card **** **** **** 7890 CVV not required for this test.",
    "My Aadhaar 2222 3333 4444 – posted by error, admin please delete.",
    "Do not misuse this PAN CFDER8901M – only for one-time use.",
    ]

    for text in samples:
        print(f"\nInput : {text}")
        results = hybrid_scan(text)
        for r in results:
            print(
                f"  [{r['severity']}] {r['type']} | score={r['risk_score']} "
                f"| intent='{r['nli_label']}'"
            )
