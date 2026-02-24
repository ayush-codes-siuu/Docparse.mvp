import json
import os

from groq import Groq

SYSTEM_PROMPT = """\
You are a document extraction assistant specialized in Indian GST Invoices.

Given raw text extracted from an invoice (via PDF text extraction or OCR), extract the following fields:

- vendor_name: The name of the seller/vendor company.
- vendor_gstin: The vendor's 15-digit GSTIN (Goods and Services Tax Identification Number).
- invoice_number: The unique invoice identifier.
- invoice_date: The date of the invoice (return in DD/MM/YYYY format when possible).
- total_taxable_value: The total taxable amount before GST (as a number).
- total_gst_amount: The total GST amount including CGST, SGST, and/or IGST (as a number).
- grand_total: The final total amount including tax (as a number).

Additionally, return a "confidence" object containing an integer score from 0 to 100 for EACH \
extracted field. The score should reflect how confident you are in the extraction based on:
- Text clarity: Was the text clearly readable or garbled/partial?
- Ambiguity: Were there multiple possible values or was the field unambiguous?
- Format match: Did the extracted value match the expected format (e.g. GSTIN pattern, date format)?

If a field is null (not found), assign a confidence of 0 for that field.

Rules:
- If a field cannot be found or is ambiguous, return null for that field.
- For numeric fields, return the number without currency symbols or commas.
- GSTIN format: 2-digit state code + 10-char PAN + 1 entity code + 1 check digit (e.g., 27AABCU9603R1ZM).
- Look for CGST + SGST or IGST to compute the total GST amount.
- Do NOT hallucinate values. Only extract what is clearly present in the text.

You MUST respond with ONLY valid JSON in exactly this structure (no markdown, no explanation):
{
  "vendor_name": "<string or null>",
  "vendor_gstin": "<string or null>",
  "invoice_number": "<string or null>",
  "invoice_date": "<string or null>",
  "total_taxable_value": <number or null>,
  "total_gst_amount": <number or null>,
  "grand_total": <number or null>,
  "confidence": {
    "vendor_name": <integer 0-100>,
    "vendor_gstin": <integer 0-100>,
    "invoice_number": <integer 0-100>,
    "invoice_date": <integer 0-100>,
    "total_taxable_value": <integer 0-100>,
    "total_gst_amount": <integer 0-100>,
    "grand_total": <integer 0-100>
  }
}"""

EXPECTED_FIELDS = [
    "vendor_name",
    "vendor_gstin",
    "invoice_number",
    "invoice_date",
    "total_taxable_value",
    "total_gst_amount",
    "grand_total",
]


def extract_invoice_fields(raw_text: str, api_key: str | None = None) -> dict:
    """Send raw invoice text to Groq (Llama 4 Scout 17B) and return structured fields.

    Args:
        raw_text: The extracted text from the invoice document.
        api_key: Groq API key. Falls back to GROQ_API_KEY env var if not provided.

    Returns:
        A dict with the 7 GST invoice fields plus a confidence object.

    Raises:
        ValueError: If no API key is available or the key is invalid.
        RuntimeError: If the API call fails.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "Groq API key is not configured. "
            "Enter it in the sidebar or set the GROQ_API_KEY environment variable."
        )

    client = Groq(api_key=key)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)

        # Validate that expected keys are present
        for field in EXPECTED_FIELDS:
            if field not in result:
                result[field] = None

        # Ensure confidence object exists with all fields
        if "confidence" not in result or not isinstance(result["confidence"], dict):
            result["confidence"] = {}
        for field in EXPECTED_FIELDS:
            if field not in result["confidence"]:
                result["confidence"][field] = 0

        return result

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from model response: {e}")
    except Exception as e:
        error_msg = str(e).lower()
        if "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
            raise ValueError("Invalid Groq API key. Please check your configuration.")
        if "rate" in error_msg and "limit" in error_msg:
            raise RuntimeError(
                "Groq rate limit reached. Please wait a moment and retry."
            )
        raise RuntimeError(f"Groq API call failed: {e}")
