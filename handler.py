"""
SmartInvoice — RunPod Serverless Handler
Wraps Qwen3-VL-8B-Instruct v6 pipeline for production invoice extraction.
"""

import runpod
import base64
import json
import re
import torch
from io import BytesIO
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

# ─────────────────────────────────────────────
# 1. MODEL LOAD  (runs once at cold start)
# ─────────────────────────────────────────────
print("[SmartInvoice] Loading Qwen3-VL-8B model...")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",          # swap to Qwen3 when available on HF
    quantization_config=bnb_cfg,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    max_pixels=1280 * 28 * 28,
)

print("[SmartInvoice] Model ready.")

# ─────────────────────────────────────────────
# 2. V6 PROMPT  (every rule earned in production)
# ─────────────────────────────────────────────
SCHEMA = """Extract exactly these 25 fields from this Tunisian invoice image and return FLAT JSON only.

SUPPLIER INFO (top/header of document):
- societe: supplier company name
- adresse: supplier address
- telephone: supplier phone number (digits only, strip trailing names)
- rc_mf: supplier fiscal IDs — MF first then space then RC (never Code Douane)

CLIENT INFO (client block in document):
- client: client company name (NOT the contact person after "A l'attention de")
- adresse_client: client address
- tva_num: client TVA/fiscal number from client block (MF or TVA N° in client zone)

DOCUMENT HEADER:
- numero_facture: invoice number (FACTURE N°, Facture Pro Forma N°, FACTURE MAGASINAGE N°)
- date: invoice date (DD/MM/YYYY)
- reference: single document-level reference only (NOT per-line BL numbers)

TOTALS (always money amounts in Tunisian Dinars):
- total_ht: total before tax
- remise: discount amount (0 if none)
- tva: TVA tax amount in dinars — NEVER a percentage
- timbre: fiscal stamp (usually 1,000)
- total_ttc: final total including all taxes
- montant_lettres: total in words (empty string "" for proformas)

LINE ITEMS — four parallel lists of EQUAL length:
- designation[]: product/service description
- quantite[]: quantity
- prix_unit[]: unit price (PUHT column)
- montant_ht[]: line total (Mnt HT column)
EXCLUDE from line items: rows that are BL numbers, Bon de livraison, Commande, Référence.

CRITICAL RULES:
1. Output FLAT JSON only. NEVER nest under SUPPLIER/CLIENT/TOTALS/LINE_ITEMS keys.
2. Comma = decimal separator. "323,500" means 323.5 dinars NOT 323500.
3. rc_mf comes from supplier header/footer. tva_num comes from client block. Never swap.
4. "Code Douane" is a customs code — NEVER put it in rc_mf.
5. tva is always a money amount (dinars), never a percentage like 19%.
6. Client is the COMPANY name, not the contact person.
7. If a field is absent or unreadable, use "" — never invent values.
8. All four line-item lists must have exactly the same number of elements.
9. Numbers: preserve original comma-decimal format e.g. "14791,850" not "14791.85"."""


# ─────────────────────────────────────────────
# 3. PIPELINE FUNCTIONS  (v6 post-processing)
# ─────────────────────────────────────────────

def extract(image: Image.Image) -> dict | None:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": SCHEMA}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=2048, do_sample=False)
    resp = processor.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    try:
        s, e = resp.find("{"), resp.rfind("}") + 1
        return json.loads(resp[s:e])
    except Exception:
        return None


def flatten_nested(p: dict) -> dict:
    """Handles Qwen grouping fields under SUPPLIER/CLIENT/TOTALS/LINE_ITEMS."""
    if not isinstance(p, dict):
        return p
    flat = {}
    for k, v in p.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[k2] = v2
        else:
            flat[k] = v
    items = (
        flat.pop("LINE_ITEMS", None)
        or flat.pop("line_items", None)
        or flat.pop("LineItems", None)
    )
    if isinstance(items, list) and items and isinstance(items[0], dict):
        flat["designation"] = [str(it.get("designation", "")) for it in items]
        flat["quantite"]    = [str(it.get("quantite",    "")) for it in items]
        flat["prix_unit"]   = [str(it.get("prix_unit",   "")) for it in items]
        flat["montant_ht"]  = [str(it.get("montant_ht",  "")) for it in items]
    return flat


REF_ROW = re.compile(
    r"^(bon\s+de\s+livraison|commande|cd\s*\d+|bl\s*\d+|b\d{4,}|r[ée]f[ée]rence|devis)",
    re.IGNORECASE,
)


def _safe_norm_num(v):
    """Parse Tunisian number format safely. Preserves 3-decimal millime precision."""
    if v is None or v == "":
        return v
    s = str(v).strip()
    cleaned = s.replace(" ", "").replace("\u00a0", "")
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) <= 3:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    try:
        float(cleaned)
        return cleaned
    except ValueError:
        return s


def post_process(p: dict) -> dict:
    if not isinstance(p, dict):
        return p

    # Normalize scalar number fields
    for field in ("total_ht", "remise", "tva", "timbre", "total_ttc",
                  "prix_unit", "montant_ht"):
        if field in p and not isinstance(p[field], list):
            p[field] = _safe_norm_num(p[field])

    # Normalize list number fields
    for field in ("prix_unit", "montant_ht", "quantite"):
        if isinstance(p.get(field), list):
            p[field] = [_safe_norm_num(x) for x in p[field]]

    # Strip trailing names from telephone
    if p.get("telephone"):
        p["telephone"] = re.split(r"\s+[A-Za-zÀ-ÿ]", str(p["telephone"]))[0].strip()

    # Filter reference rows from line items
    if isinstance(p.get("designation"), list):
        keep_indices = [
            i for i, d in enumerate(p["designation"])
            if not REF_ROW.match(str(d).strip())
        ]
        for field in ("designation", "quantite", "prix_unit", "montant_ht"):
            if isinstance(p.get(field), list):
                p[field] = [p[field][i] for i in keep_indices if i < len(p[field])]

    return p


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


def validate(p: dict) -> tuple[bool, list[str]]:
    issues = []

    # Critical fields must be present
    for f in ("client", "numero_facture", "date", "total_ttc"):
        if not p.get(f):
            issues.append(f"missing critical field: {f}")

    ht  = _to_float(p.get("total_ht"))
    rem = _to_float(p.get("remise")) or 0.0
    tva = _to_float(p.get("tva"))
    tmb = _to_float(p.get("timbre")) or 0.0
    ttc = _to_float(p.get("total_ttc"))

    # Math check
    if ht and tva and ttc:
        expected = ht - rem + tva + tmb
        if abs(expected - ttc) > 1.0:
            issues.append(f"math mismatch: {ht}-{rem}+{tva}+{tmb}={expected:.3f} ≠ ttc={ttc}")

    # Magnitude sanity (Tunisian comma-as-decimal misread catches)
    for label, val in [("total_ht", ht), ("total_ttc", ttc), ("tva", tva)]:
        if val and val >= 1_000_000:
            issues.append(f"magnitude suspect: {label}={val} (≥1M, likely 1000x misread)")
    if tmb and tmb >= 100:
        issues.append(f"timbre={tmb} suspicious (≥100)")

    # rc_mf == tva_num collision
    rc, tn = (p.get("rc_mf") or "").strip(), (p.get("tva_num") or "").strip()
    if rc and tn and rc == tn:
        issues.append("rc_mf identical to tva_num — likely a swap")

    # tva equals total_ttc (proforma misread)
    if tva and ttc and abs(tva - ttc) < 1.0:
        issues.append(f"tva ({tva}) equals total_ttc ({ttc}) — likely misread")

    # tva > total_ht (impossible)
    if tva and ht and tva > ht:
        issues.append(f"tva ({tva}) > total_ht ({ht}) — suspicious")

    # Line-item list length mismatch
    lens = [len(p.get(f, [])) for f in ("designation", "quantite", "prix_unit", "montant_ht")]
    if len(set(lens)) > 1:
        issues.append(f"line-item list lengths mismatch: {lens}")

    return (len(issues) == 0), issues


def process_invoice(image: Image.Image) -> dict:
    raw = extract(image)
    if raw is None:
        return {"data": {}, "auto_accept": False, "issues": ["extraction_failed: could not parse JSON from model output"]}
    raw     = flatten_nested(raw)
    cleaned = post_process(raw)
    ok, issues = validate(cleaned)
    return {"data": cleaned, "auto_accept": ok, "issues": issues}


# ─────────────────────────────────────────────
# 4. RUNPOD HANDLER ENTRY POINT
# ─────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    Expected input:
      {
        "input": {
          "image_base64": "<base64-encoded PNG or JPEG>"
        }
      }
    Returns:
      {
        "data": { ...25 fields... },
        "auto_accept": true/false,
        "issues": [ ...list of validation issues... ]
      }
    """
    try:
        job_input = job.get("input", {})
        image_b64 = job_input.get("image_base64")

        if not image_b64:
            return {"error": "Missing 'image_base64' in input"}

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        result = process_invoice(image)
        return result

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
