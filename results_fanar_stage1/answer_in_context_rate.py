"""
Computes the percentage of MCQ questions where the correct answer text
is directly present in the context (string match) or semantically similar
(token overlap / normalized substring).

Usage:
    python answer_in_context_rate.py --input your_file.json
"""

import json
import argparse
import re
import unicodedata
from pathlib import Path


# ──────────────────────────────────────────────
# Arabic text normalization
# ──────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fair comparison."""
    if not text:
        return ""
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)
    # Remove diacritics (tashkeel)
    text = re.sub(r"[\u0617-\u061A\u064B-\u065F]", "", text)
    # Normalize alef variants → bare alef
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize teh marbuta → heh
    text = re.sub(r"ة", "ه", text)
    # Normalize yeh variants
    text = re.sub(r"[يى]", "ي", text)
    # Remove tatweel
    text = text.replace("ـ", "")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set:
    """Simple whitespace tokenizer returning a set of tokens."""
    return set(normalize_arabic(text).split())


# ──────────────────────────────────────────────
# Match strategies
# ──────────────────────────────────────────────

def exact_match(answer: str, context: str) -> bool:
    """Normalized exact substring match."""
    return normalize_arabic(answer) in normalize_arabic(context)


def token_overlap(answer: str, context: str, threshold: float = 0.8) -> bool:
    """
    Fraction of answer tokens found in context tokens.
    Returns True if overlap >= threshold.
    """
    a_tokens = tokenize(answer)
    c_tokens = tokenize(context)
    if not a_tokens:
        return False
    overlap = len(a_tokens & c_tokens) / len(a_tokens)
    return overlap >= threshold


# ──────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────

def analyse(data: list, token_threshold: float = 0.8) -> dict:
    total = 0
    exact_hits = 0
    token_hits = 0
    skipped = 0  # items without context or correct_text

    per_item = []

    for item in data:
        context = item.get("context", "").strip()
        # Prefer correct_text if available; fall back to options[correct_answer]
        correct_text = item.get("correct_text", "").strip()
        if not correct_text:
            key = item.get("correct_answer", "")
            correct_text = item.get("options", {}).get(key, "").strip()

        if not context or not correct_text:
            skipped += 1
            continue

        total += 1
        em = exact_match(correct_text, context)
        to = token_overlap(correct_text, context, threshold=token_threshold)

        if em:
            exact_hits += 1
        if to:
            token_hits += 1

        per_item.append({
            "question": item.get("question", "")[:80],
            "correct_text": correct_text,
            "exact_match": em,
            "token_overlap": to,
        })

    exact_pct = 100 * exact_hits / total if total else 0
    token_pct = 100 * token_hits / total if total else 0

    return {
        "total_items": total,
        "skipped": skipped,
        "exact_match_count": exact_hits,
        "exact_match_pct": round(exact_pct, 2),
        "token_overlap_count": token_hits,
        "token_overlap_pct": round(token_pct, 2),
        "token_threshold_used": token_threshold,
        "per_item": per_item,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Answer-in-context rate for Arabic MCQ JSON")
    parser.add_argument("--input", required=True, help="Path to JSON file")
    parser.add_argument("--token_threshold", type=float, default=0.8,
                        help="Min token overlap fraction to count as 'present' (default 0.8)")
    parser.add_argument("--show_false_negatives", action="store_true",
                        help="Print items where exact match fails but token overlap passes")
    parser.add_argument("--save_report", help="Optional path to save per-item CSV report")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERROR] JSON root must be a list of MCQ items.")
        return

    print(f"\nLoaded {len(data)} items from {path.name}")
    results = analyse(data, token_threshold=args.token_threshold)

    print("\n" + "=" * 55)
    print("  ANSWER-IN-CONTEXT RATE REPORT")
    print("=" * 55)
    print(f"  Total items analysed : {results['total_items']}")
    print(f"  Skipped (no ctx/ans) : {results['skipped']}")
    print(f"\n  Exact substring match: {results['exact_match_count']} "
          f"({results['exact_match_pct']}%)")
    print(f"  Token overlap ≥{int(args.token_threshold*100)}%  : {results['token_overlap_count']} "
          f"({results['token_overlap_pct']}%)")
    print("=" * 55)

    # Useful for paper: the exact match % is what goes in the TODO
    print(f"\n  → For your paper TODO: exact match = {results['exact_match_pct']}%")
    print(f"  → Soft match (token overlap ≥{int(args.token_threshold*100)}%) = {results['token_overlap_pct']}%")

    if args.show_false_negatives:
        fn = [x for x in results["per_item"]
              if not x["exact_match"] and x["token_overlap"]]
        print(f"\n  Items where exact match fails but token overlap passes: {len(fn)}")
        for x in fn[:10]:
            print(f"    Q: {x['question']}")
            print(f"    A: {x['correct_text']}\n")

    if args.save_report:
        import csv
        with open(args.save_report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "correct_text",
                                                    "exact_match", "token_overlap"])
            writer.writeheader()
            writer.writerows(results["per_item"])
        print(f"\n  Per-item report saved to: {args.save_report}")


if __name__ == "__main__":
    main()