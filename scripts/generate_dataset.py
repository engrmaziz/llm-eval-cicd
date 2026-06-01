"""Generate a synthetic golden dataset for DeepEval.

This script uses the google-generativeai SDK with gemini-1.5-flash to
create 100 customer-support question-answer-context trios and writes the
final array to tests/dataset.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai


TOTAL_ITEMS = 100
BATCH_SIZE = 5
MODEL_NAME = "gemini-1.5-flash"
SLEEP_SECONDS = 5

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "tests" / "dataset.json"

KNOWLEDGE_BASE = """
Customer Support Knowledge Base

1. Returns and refunds
- Opened items may be returned within 30 days of delivery.
- After 30 days, opened items are not eligible for a refund, but some orders may qualify for store credit.
- Unopened items may be returned within 60 days of delivery.
- Refunds are issued to the original payment method within 5 to 10 business days after approval.
- Return shipping is free only for defective items or warehouse fulfillment errors.

2. Shipping and delivery
- Standard shipping takes 3 to 5 business days.
- Expedited shipping takes 1 to 2 business days.
- Orders placed after 2:00 PM local time ship the next business day.
- Tracking numbers are emailed when the package is scanned by the carrier.
- PO boxes cannot receive expedited shipments.

3. Warranty and repairs
- Electronics include a 1-year limited warranty.
- Accessories include a 90-day limited warranty.
- Warranty claims require proof of purchase.
- Physical damage, liquid damage, and unauthorized repairs are not covered.
- Repair turnaround is usually 7 to 14 business days after the item is received.

4. Account and payments
- Customers can update their email address from the account settings page.
- Only one promo code can be used per order.
- Gift cards cannot be combined with subscription purchases.
- Failed payments are retried up to three times over 48 hours.
- VAT invoices are available for business customers in supported regions.

5. Support policies
- Live chat is available daily from 8:00 AM to 8:00 PM UTC.
- Email support replies within 1 to 2 business days.
- Damaged delivery claims must be submitted within 48 hours of delivery.
- Missing item claims must include the order number and a photo of the package label.
- Cancellations are only possible before the order is packed for shipping.
""".strip()


def configure_model() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running this script.")
    genai.configure(api_key=api_key)


def build_prompt(batch_size: int, existing_inputs: List[str]) -> str:
    existing_examples = "\n".join(f"- {item}" for item in existing_inputs[-20:]) or "- None"
    return f"""
You are generating a synthetic golden dataset for customer support evaluation.

Use only the knowledge base facts below. Do not invent policies beyond the facts.
Return valid JSON only, with no markdown fences and no commentary.

Target schema for each object:
{{
  "input": "string",
  "expected_output": "string",
  "context": ["string", "string"]
}}

Requirements:
- Produce exactly {batch_size} unique items.
- The domain must be customer support for e-commerce policies and tech hardware.
- Make the questions varied: refunds, shipping, warranties, payments, claims, cancellations, and support hours.
- Each context array must contain 1 to 3 short factual strings from the knowledge base.
- Keep expected_output concise, grounded, and directly answer the question.
- Avoid repeating any of these recently generated inputs:
{existing_examples}

Knowledge base:
{KNOWLEDGE_BASE}
""".strip()


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON array.")

    payload = text[start : end + 1]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Model response JSON must be an array.")
    return data


def validate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each dataset item must be an object.")

    required_keys = {"input", "expected_output", "context"}
    if set(item) != required_keys:
        raise ValueError(f"Dataset item must contain only {sorted(required_keys)}.")

    input_text = item["input"]
    expected_output = item["expected_output"]
    context = item["context"]

    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("'input' must be a non-empty string.")
    if not isinstance(expected_output, str) or not expected_output.strip():
        raise ValueError("'expected_output' must be a non-empty string.")
    if not isinstance(context, list) or not context:
        raise ValueError("'context' must be a non-empty list of strings.")
    if not all(isinstance(entry, str) and entry.strip() for entry in context):
        raise ValueError("Every context entry must be a non-empty string.")

    return {
        "input": input_text.strip(),
        "expected_output": expected_output.strip(),
        "context": [entry.strip() for entry in context],
    }


def generate_batch(model: Any, batch_size: int, existing_inputs: List[str]) -> List[Dict[str, Any]]:
    prompt = build_prompt(batch_size, existing_inputs)
    response = model.generate_content(prompt)
    response_text = getattr(response, "text", None) or str(response)
    raw_items = extract_json_array(response_text)
    return [validate_item(item) for item in raw_items]


def generate_dataset() -> List[Dict[str, Any]]:
    configure_model()
    model = genai.GenerativeModel(MODEL_NAME)

    items: List[Dict[str, Any]] = []
    seen_inputs = set()

    print(f"🚀 Starting synthetic dataset generation for {TOTAL_ITEMS} items...")

    while len(items) < TOTAL_ITEMS:
        remaining = TOTAL_ITEMS - len(items)
        batch_size = min(BATCH_SIZE, remaining)

        try:
            batch_items = generate_batch(model, batch_size, [item["input"] for item in items])

            for item in batch_items:
                if item["input"] in seen_inputs:
                    continue
                seen_inputs.add(item["input"])
                items.append(item)
                if len(items) == TOTAL_ITEMS:
                    break

            print(f"📦 Progress: {len(items)}/{TOTAL_ITEMS} items successfully compiled.")

        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Warning: Batch generation failed due to formatting issues ({e}). Retrying...")
            time.sleep(SLEEP_SECONDS)
            continue

        if len(items) < TOTAL_ITEMS:
            time.sleep(SLEEP_SECONDS)

    return items


def save_dataset(dataset: List[Dict[str, Any]]) -> None:
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    dataset = generate_dataset()
    save_dataset(dataset)
    print(f"✅ Success! Wrote {len(dataset)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()