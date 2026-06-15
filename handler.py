import os, re, json, traceback, base64, io
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import runpod

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Loading model...")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LOAD_ERROR = None
model = None
processor = None

try:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL,
        max_pixels=1280 * 28 * 28,
        cache_dir=os.environ.get("HF_HOME", None)
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME", None)
    )
    print("Model loaded successfully.")
except Exception:
    LOAD_ERROR = traceback.format_exc()
    print("Model load FAILED:\n", LOAD_ERROR)

SCHEMA = """You extract structured data from Tunisian business invoices (French + Arabic).
Return ONLY a valid JSON object — no prose, no markdown.
Output a FLAT JSON object with the exact field names listed below — DO NOT nest fields
under groups like "SUPPLIER" or "CLIENT" or "TOTALS". All fields go at the top level.
For every field, output VALUES only, never label text.
Use "" if a field is genuinely absent from the invoice. Never guess.

TUNISIAN NUMBER FORMAT (READ THIS CAREFULLY):
Tunisian invoices use COMMA as the decimal separator and SPACE (or nothing) as thousands separator.
  "323,500"     means 323 dinars and 500 millimes — NOT three hundred thousand
  "1 234,567"   means 1234 dinars and 567 millimes — NOT one million
  "15 117,154"  means 15117 dinars and 154 millimes
  "1,000"       means 1 dinar — the timbre is always 1,000 (= 1 dinar)
Most Tunisian invoice totals are between 10 dinars and 100,000 dinars.
Output numbers EXACTLY as written on the invoice (keep the comma).

FIELDS (all at the TOP level of the JSON):

SUPPLIER info (top of invoice or footer):
- societe: company name
- adresse: supplier postal address
- telephone: digits and standard separators only, strip trailing names or labels
- email: supplier email
- rc_mf: supplier fiscal identifiers from footer. Format: "<MF> <RC>" with single space. NEVER from client block.
- rib_banque: supplier bank account (RIB / Compte Bancaire / IBAN)

CLIENT info (in the customer block):
- client: customer COMPANY name
- code_client: customer code (CODE CLIENT)
- adresse_client: customer postal address
- tva_num: customer fiscal/TVA number from client block only

INVOICE METADATA:
- numero_facture: invoice number
- date: invoice date in DD/MM/YYYY format
- mode_paiement: payment method

TOTALS:
- total_ht: subtotal before tax
- remise: discount amount in dinars ("" if shown only as percentage)
- tva: TAX AMOUNT in dinars (NEVER the percentage rate like 19%)
- timbre: stamp duty, usually "1,000"
- total_ttc: grand total (NET A PAYER, Total TTC)
- montant_lettres: amount written in words

LINE ITEMS — four parallel lists of equal length:
- designation[]: product description
- quantite[]: quantity
- prix_unit[]: unit price
- montant_ht[]: line total

CRITICAL RULES:
1. FLAT JSON only. No nested objects.
2. Numbers: comma is decimal separator. "323,500" is 323.5 dinars.
3. rc_mf comes from supplier footer. tva_num comes from client block. Never swap.
4. tva is always a money amount, never a percentage.
5. If absent, return "" not null."""


def flatten_nested(data):
    if not isinstance(data, dict):
        return data
    flat = {}
    nested_keys = {"SUPPLIER", "CLIENT", "TOTALS", "ITEMS", "HEADER",
                   "supplier", "client_info", "totals", "items", "header",
                   "line_items", "invoice_details", "INVOICE_METADATA"}
    for k, v in data.items():
        if k.upper() in {nk.upper() for nk in nested_keys} and isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    items = (flat.pop("LINE_ITEMS", None) or flat.pop("line_items", None)
             or flat.pop("LineItems", None))
    if isinstance(items, list) and items and isinstance(items[0], dict):
        flat["designation"] = [str(it.get("designation", "")) for it in items]
        flat["quantite"]    = [str(it.get("quantite", "")) for it in items]
        flat["prix_unit"]   = [str(it.get("prix_unit", "")) for it in items]
        flat["montant_ht"]  = [str(it.get("montant_ht", "")) for it in items]
    return flat


def _to_num(s):
    try:
        return float(str(s).replace(",", ".").replace(" ", "").replace("\u00a0", ""))
    except Exception:
        return None


def validate(data):
    issues = []
    ht  = _to_num(data.get("total_ht"))
    rem = _to_num(data.get("remise")) or 0
    tva = _to_num(data.get("tva")) or 0
    tim = _to_num(data.get("timbre")) or 0
    ttc = _to_num(data.get("total_ttc"))

    if ht is not None and ttc is not None:
        expected = ht - rem + tva + tim
        if abs(expected - ttc) > 1.0:
            issues.append(f"math_mismatch: {ht}-{rem}+{tva}+{tim}={expected:.3f} != ttc={ttc}")

    suspicious = [v for v in (ht, ttc) if v is not None and v >= 1_000_000]
    if suspicious or (tim and tim >= 100):
        issues.append(f"magnitude_error: ht={ht}, ttc={ttc}, timbre={tim}")

    date = data.get("date", "")
    if date and not re.match(r"\d{2}/\d{2}/\d{4}$", str(date)):
        issues.append(f"date_format: '{date}'")

    for f in ("client", "numero_facture", "date", "total_ttc"):
        if not data.get(f):
            issues.append(f"missing: {f}")

    rc = (data.get("rc_mf") or "").strip()
    tn = (data.get("tva_num") or "").strip()
    if rc and tn and rc == tn:
        issues.append("rc_mf identical to tva_num — likely swapped")

    lens = [len(data.get(f, [])) for f in ("designation", "quantite", "prix_unit", "montant_ht")
            if isinstance(data.get(f), list)]
    if len(set(lens)) > 1:
        issues.append(f"line_item_length_mismatch: {lens}")

    return len(issues) == 0, issues


def run_inference(image: Image.Image) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": SCHEMA}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return response


def process_invoice(image: Image.Image) -> dict:
    raw_text = run_inference(image)
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s, e = text.find("{"), text.rfind("}") + 1
    if s < 0 or e <= s:
        return {"error": "no_json_found", "raw": raw_text[:500]}
    try:
        data = json.loads(text[s:e])
    except json.JSONDecodeError as ex:
        return {"error": f"json_parse_failed: {ex}", "raw": raw_text[:500]}
    data = flatten_nested(data)
    auto_accept, issues = validate(data)
    return {"data": data, "auto_accept": auto_accept, "issues": issues}


def handler(job):
    if LOAD_ERROR:
        return {"error": f"model_not_loaded: {LOAD_ERROR[:300]}"}

    job_input = job.get("input", {})
    image_b64 = job_input.get("image_base64") or job_input.get("image")
    if not image_b64:
        return {"error": "missing image_base64 in input"}

    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as ex:
        return {"error": f"image_decode_failed: {ex}"}

    w, h = image.size
    if max(w, h) > 1600:
        ratio = 1600 / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    try:
        result = process_invoice(image)
    except Exception:
        return {"error": traceback.format_exc()}

    return result


runpod.serverless.start({"handler": handler})
