from langchain_core.messages import HumanMessage
from app.services.llm_service import get_llm
from app.services.pdf_service import PageImage
from app.utils.helpers import safe_parse_json

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

CLASSIFICATION_PROMPT = """You are a medical insurance document classifier.

Analyze this page image and classify it into EXACTLY ONE of these types:
- claim_forms: Insurance claim application forms
- cheque_or_bank_details: Bank cheques, account details, NEFT forms
- identity_document: Aadhaar, PAN, passport, driving license, policy docs
- itemized_bill: Hospital bills with line items and costs
- discharge_summary: Hospital discharge reports with diagnosis
- prescription: Doctor prescriptions for medicines
- investigation_report: Lab reports, X-rays, diagnostic test results
- cash_receipt: Payment receipts, transaction confirmations
- other: Anything that doesn't fit above categories

Respond with ONLY a JSON object in this exact format:
{
    "document_type": "<one of the types above>",
    "confidence": <float between 0 and 1>,
    "reasoning": "<brief one-line explanation>"
}

Do NOT include any other text outside the JSON."""


class SegregatorAgent:
    """Classifies each PDF page into a document type using NVIDIA NIM vision."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

    def classify_page(self, page: PageImage) -> dict:
        """Classify a single page image."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": CLASSIFICATION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page.base64_image}",
                    },
                },
            ]
        )
        response = self.llm.invoke([message])
        result = safe_parse_json(response.content)

        if result["document_type"] not in DOCUMENT_TYPES:
            result["document_type"] = "other"

        result["page_number"] = page.page_number
        return result

    def classify_all_pages(
        self, pages: list[PageImage]
    ) -> dict:
        """Classify all pages and return a routing map.

        Returns:
            {
                "classifications": [
                    {"page_number": 1, "document_type": "...", ...},
                    ...
                ],
                "routing": {
                    "identity_document": [1, 3],
                    "discharge_summary": [2],
                    "itemized_bill": [4, 5],
                    ...
                }
            }
        """
        classifications = []
        routing: dict[str, list[int]] = {}

        for page in pages:
            result = self.classify_page(page)
            classifications.append(result)

            doc_type = result["document_type"]
            if doc_type not in routing:
                routing[doc_type] = []
            routing[doc_type].append(page.page_number)

        return {
            "classifications": classifications,
            "routing": routing,
        }
