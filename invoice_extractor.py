"""
Invoice OCR Extraction System
Pipeline A: Tesseract OCR + Regex Rules
Pipeline B: OpenCV Preprocessing + Positional Field Detection
"""

import os
import re
import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from pathlib import Path
from difflib import SequenceMatcher

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_DIR = Path("/mnt/user-data/uploads")  # adjust if needed
OUTPUT_CSV = "/mnt/user-data/outputs/output.csv"
COMPARISON_CSV = "/mnt/user-data/outputs/comparison_report.csv"

FIELDS = [
    "seller_name", "seller_tax_id", "client_name", "client_tax_id",
    "invoice_number", "invoice_date", "net_worth", "vat", "gross_worth"
]

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────
PATTERNS = {
    "invoice_number": [
        r"(?:invoice\s*(?:no|number|#|num)[:\s#]*)([\w\-/]+)",
        r"(?:inv[.\s]*no[:\s]*)([\w\-/]+)",
        r"(?:^|\s)(INV[-\s]?\d{4,})",
        r"invoice\s*:\s*([\w\-]+)",
    ],
    "invoice_date": [
        r"(?:invoice\s*date|date\s*of\s*invoice|date)[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(?:date)[:\s]*(\d{1,2}\s+\w+\s+\d{4})",
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        r"(\d{4}-\d{2}-\d{2})",
    ],
    "seller_tax_id": [
        r"(?:seller|vendor|from|our)[^\n]*(?:tax\s*id|vat\s*no|tin|gst|cif|nip)[:\s]*([A-Z0-9\-]{6,20})",
        r"(?:tax\s*id|vat\s*reg|tin|nip|cif)[:\s]*([A-Z0-9\-]{6,20})",
        r"(?:vat\s*number)[:\s]*([A-Z0-9\-]{6,20})",
    ],
    "client_tax_id": [
        r"(?:client|customer|buyer|bill\s*to|to)[^\n]*(?:tax\s*id|vat\s*no|tin|nip|cif)[:\s]*([A-Z0-9\-]{6,20})",
        r"(?:customer\s*tax|client\s*vat|buyer\s*tin)[:\s]*([A-Z0-9\-]{6,20})",
    ],
    "net_worth": [
        r"(?:net\s*(?:amount|worth|total|value|price)|subtotal|sub\s*total)[:\s]*\$?([\d,]+\.?\d*)",
        r"(?:total\s*(?:before|excl|net))[:\s]*\$?([\d,]+\.?\d*)",
        r"(?:amount\s*before\s*tax)[:\s]*\$?([\d,]+\.?\d*)",
    ],
    "vat": [
        r"(?:vat|tax\s*amount|gst|tax)[:\s]*\$?([\d,]+\.?\d*)",
        r"(?:vat\s*\d+%?)[:\s]*\$?([\d,]+\.?\d*)",
        r"(?:sales\s*tax)[:\s]*\$?([\d,]+\.?\d*)",
    ],
    "gross_worth": [
        r"(?:gross|total\s*(?:amount|due|payable)|amount\s*due|total\s*invoice)[:\s]*\$?([\d,]+\.?\d*)",
        r"(?:grand\s*total|invoice\s*total|total\s*with\s*tax)[:\s]*\$?([\d,]+\.?\d*)",
        r"(?:balance\s*due|please\s*pay)[:\s]*\$?([\d,]+\.?\d*)",
    ],
}

NAME_KEYWORDS = {
    "seller": ["from:", "seller:", "vendor:", "billed by:", "issued by:", "supplier:", "company:"],
    "client": ["to:", "bill to:", "client:", "customer:", "buyer:", "ship to:"],
}


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def clean_amount(val):
    if val:
        return val.replace(",", "").strip()
    return ""


def apply_patterns(text, field):
    patterns = PATTERNS.get(field, [])
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return ""


def extract_name_near_keyword(lines, keyword_list, max_lookahead=3):
    """Find a name on the same line or following lines after a keyword."""
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        for kw in keyword_list:
            if kw in ll:
                # Check same line remainder
                idx = ll.index(kw) + len(kw)
                remainder = line[idx:].strip().strip(":").strip()
                if remainder and len(remainder) > 2:
                    return remainder
                # Check next lines
                for j in range(1, max_lookahead + 1):
                    if i + j < len(lines):
                        nxt = lines[i + j].strip()
                        if nxt and len(nxt) > 2 and not any(
                            kw2 in nxt.lower() for kw2 in ["tax", "vat", "address", "street", "city", "phone"]
                        ):
                            return nxt
    return ""


def similarity(a, b):
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio(), 3)


def match_status(a, b, threshold=0.85):
    if not a and not b:
        return "both_empty"
    if not a or not b:
        return "one_missing"
    s = similarity(a, b)
    return "match" if s >= threshold else "mismatch"


# ─────────────────────────────────────────────
# PIPELINE A: Raw Tesseract + Regex
# ─────────────────────────────────────────────
def pipeline_a(img_path):
    img = Image.open(img_path).convert("RGB")
    raw = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    text = raw
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    flat = " ".join(lines)

    result = {}
    for field in ["invoice_number", "invoice_date", "seller_tax_id", "client_tax_id",
                  "net_worth", "vat", "gross_worth"]:
        result[field] = apply_patterns(text, field)

    result["net_worth"] = clean_amount(result["net_worth"])
    result["vat"] = clean_amount(result["vat"])
    result["gross_worth"] = clean_amount(result["gross_worth"])

    result["seller_name"] = extract_name_near_keyword(lines, NAME_KEYWORDS["seller"])
    result["client_name"] = extract_name_near_keyword(lines, NAME_KEYWORDS["client"])
    return result


# ─────────────────────────────────────────────
# PIPELINE B: OpenCV Preprocessing + Positional
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Deskew + denoise + binarize
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Upscale if small
    h, w = binary.shape
    if w < 1200:
        binary = cv2.resize(binary, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return binary


def pipeline_b(img_path):
    proc = preprocess_image(img_path)
    # Use different PSM for structured layout
    config = "--oem 3 --psm 4 -c preserve_interword_spaces=1"
    raw = pytesseract.image_to_string(proc, config=config)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    text = raw

    result = {}

    # Field extraction with slightly different pattern priority
    for field in ["invoice_number", "invoice_date", "seller_tax_id", "client_tax_id",
                  "net_worth", "vat", "gross_worth"]:
        result[field] = apply_patterns(text, field)

    result["net_worth"] = clean_amount(result["net_worth"])
    result["vat"] = clean_amount(result["vat"])
    result["gross_worth"] = clean_amount(result["gross_worth"])

    # Use top-third of image lines for seller, bottom-third for client bias
    top_lines = lines[:max(1, len(lines) // 3)]
    mid_lines = lines[len(lines) // 3: 2 * len(lines) // 3]

    result["seller_name"] = extract_name_near_keyword(top_lines, NAME_KEYWORDS["seller"])
    if not result["seller_name"]:
        result["seller_name"] = extract_name_near_keyword(lines, NAME_KEYWORDS["seller"])

    result["client_name"] = extract_name_near_keyword(mid_lines, NAME_KEYWORDS["client"])
    if not result["client_name"]:
        result["client_name"] = extract_name_near_keyword(lines, NAME_KEYWORDS["client"])

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def find_images(base_dir):
    """Find images in the specified batch folder structure."""
    # Try common paths
    candidates = [
        base_dir / "batch_1" / "batch_1" / "batch1_1",
        base_dir / "batch_1" / "batch1_1",
        base_dir / "batch1_1",
        base_dir,
    ]
    for d in candidates:
        if d.exists():
            imgs = sorted(
                [f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}],
                key=lambda x: x.name
            )
            # Filter batch1-0331 to batch1-0381
            target = [f for f in imgs if any(
                f.name.startswith(f"batch1-0{i}") or f.stem.endswith(str(i))
                for i in range(331, 382)
            )]
            if target:
                return target
            if imgs:
                return imgs[:50]  # fallback: first 50
    return []


def run():
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    images = find_images(IMAGE_DIR)

    if not images:
        # Try flat directory
        all_imgs = sorted([
            f for f in IMAGE_DIR.rglob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])[:50]
        images = all_imgs

    print(f"Found {len(images)} images")

    rows_a, rows_b, comparison = [], [], []

    for img_path in images:
        name = img_path.name
        print(f"  Processing {name} ...", end=" ")

        try:
            ra = pipeline_a(img_path)
        except Exception as e:
            ra = {f: "" for f in FIELDS}
            print(f"[PipeA ERROR: {e}]", end=" ")

        try:
            rb = pipeline_b(img_path)
        except Exception as e:
            rb = {f: "" for f in FIELDS}
            print(f"[PipeB ERROR: {e}]", end=" ")

        # Merge: prefer non-empty value; tie-break → Pipeline A
        merged = {}
        for f in FIELDS:
            merged[f] = ra.get(f, "") or rb.get(f, "")

        rows_a.append({"image": name, **{f: ra.get(f, "") for f in FIELDS}})
        rows_b.append({"image": name, **{f: rb.get(f, "") for f in FIELDS}})

        # Comparison row
        comp_row = {"image": name}
        for f in FIELDS:
            va, vb = ra.get(f, ""), rb.get(f, "")
            comp_row[f"A_{f}"] = va
            comp_row[f"B_{f}"] = vb
            comp_row[f"sim_{f}"] = similarity(va, vb)
            comp_row[f"status_{f}"] = match_status(va, vb)
        comparison.append(comp_row)
        print("done")

    # Build final output.csv (merged best values)
    out_rows = []
    for ra, rb in zip(rows_a, rows_b):
        row = {"image": ra["image"]}
        for f in FIELDS:
            row[f] = ra[f] or rb[f]
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows, columns=["image"] + FIELDS)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved output.csv → {OUTPUT_CSV}  ({len(df_out)} rows)")

    df_cmp = pd.DataFrame(comparison)
    df_cmp.to_csv(COMPARISON_CSV, index=False)
    print(f"Saved comparison_report.csv → {COMPARISON_CSV}  ({len(df_cmp)} rows)")

    # Summary stats
    print("\n── Field Fill Rates (merged output) ──")
    for f in FIELDS:
        filled = df_out[f].astype(bool).sum()
        print(f"  {f:<20} {filled}/{len(df_out)} ({100*filled//len(df_out)}%)")

    print("\n── Pipeline Agreement Rate ──")
    status_cols = [c for c in df_cmp.columns if c.startswith("status_")]
    for sc in status_cols:
        match = (df_cmp[sc] == "match").sum()
        print(f"  {sc:<30} {match}/{len(df_cmp)} match")


if __name__ == "__main__":
    run()
