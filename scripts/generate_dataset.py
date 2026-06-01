"""Generate a synthetic golden dataset for DeepEval using Groq.

This script uses the groq SDK with llama-3.3-70b-versatile and native
JSON Mode to reliably create 100 customer-support question-answer-context
trios and writes the final array to tests/dataset.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq


TOTAL_ITEMS = 100
BATCH_SIZE = 5
MODEL_NAME = "llama-3.3-70b-versatile"
SLEEP_SECONDS = 3  # Stable delay for Groq request rate compliance

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


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running this script.")
    return Groq(api_key=api_key)


def build_prompt(batch_size: int, existing_inputs: List[str]) -> str:
    existing_examples = "\n".join(f"- {item}" for item in existing_inputs[-20:]) or "- None"
    return f"""
You are generating a synthetic golden dataset for customer support evaluation.

Use only the knowledge base facts below. Do not invent policies or rules beyond the facts.
You must return a valid JSON object containing a "dataset" key which holds an array of exactly {batch_size} items.

Target JSON Object structure:
{{
  "dataset": [
    {{
      "input": "string",
      "expected_output": "string",
      "context": ["string", "string"]
    }}
  ]
}}

Requirements:
- Produce exactly {batch_size} unique items inside the "dataset" array wrapper.
- The domain must be customer support for e-commerce policies and tech hardware.
- Make the questions highly varied: refunds, shipping, warranties, payments, claims, cancellations, and support hours.
- Each context array must contain 1 to 3 short factual strings exactly matching the knowledge base.
- Keep expected_output concise, grounded, and directly answering the question.
- Avoid repeating or duplicating any of these recently generated inputs:
{existing_examples}

Knowledge base:
{KNOWLEDGE_BASE}
""".strip()


def validate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each dataset item must be an object.")

    required_keys = {"input", "expected_output", "context"}
    if not required_keys.issubset(set(item)):
        raise ValueError(f"Dataset item is missing required keys: {required_keys}")

    input_text = str(item.get("input", "")).strip()
    expected_output = str(item.get("expected_output", "")).strip()
    context = item.get("context", [])

    if not isinstance(context, list):
        context = [context]

    clean_context = [str(entry).strip() for entry in context if str(entry).strip()]

    if not input_text or not expected_output or not clean_context:
        raise ValueError("Dataset item fields cannot be empty.")

    return {
        "input": input_text,
        "expected_output": expected_output,
        "context": clean_context,
    }


def generate_batch(client: Groq, batch_size: int, existing_inputs: List[str]) -> List[Dict[str, Any]]:
    prompt = build_prompt(batch_size, existing_inputs)
    
    # Leveraging Groq's native JSON mode to guarantee syntax validity
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    response_text = response.choices[0].message.content
    data = json.loads(response_text)
    
    if "dataset" not in data or not isinstance(data["dataset"], list):
        raise ValueError("Model response JSON did not contain a 'dataset' key array.")
        
    return [validate_item(item) for item in data["dataset"]]


def generate_dataset() -> List[Dict[str, Any]]:
    client = get_client()
    items: List[Dict[str, Any]] = []
    seen_inputs = set()

    print(f"🚀 Starting Groq dataset generation for {TOTAL_ITEMS} items...")

    while len(items) < TOTAL_ITEMS:
        remaining = TOTAL_ITEMS - len(items)
        batch_size = min(BATCH_SIZE, remaining)

        try:
            batch_items = generate_batch(client, batch_size, [item["input"] for item in items])

            for item in batch_items:
                if item["input"] in seen_inputs:
                    continue
                seen_inputs.add(item["input"])
                items.append(item)
                if len(items) == TOTAL_ITEMS:
                    break

            print(f"📦 Progress: {len(items)}/{TOTAL_ITEMS} items successfully compiled.")

        except Exception as e:
            print(f"⚠️ Warning: Batch generation encountered an error ({e}). Retrying in 5s...")
            time.sleep(5)
            continue

        if len(items) < TOTAL_ITEMS:
            time.sleep(SLEEP_SECONDS)

    return items


def save_dataset(dataset: List[Dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    dataset = generate_dataset()
    save_dataset(dataset)
    print(f"✅ Success! Wrote {len(dataset)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()