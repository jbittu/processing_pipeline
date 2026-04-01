from langchain_core.messages import HumanMessage
from app.services.llm_service import get_llm
from app.services.pdf_service import PageImage
from app.models.schemas import ItemizedBillInfo
from app.utils.helpers import safe_parse_json
from tenacity import retry, stop_after_attempt, wait_exponential

EXTRACTION_PROMPT = """You are a medical billing data extractor.
You are given page(s) from a hospital itemized bill.

Extract ALL of the following fields. If a field is not found, set it to null.

Required fields:
- hospital_name: Name of the hospital
- bill_number: Bill/invoice number
- bill_date: Date of the bill (YYYY-MM-DD format if possible)
- patient_name: Patient name on the bill
- items: Array of line items, each with:
    - description: Item/service description
    - quantity: Number of units (default 1)
    - unit_price: Price per unit (null if not shown separately)
    - amount: Total amount for this line item
- subtotal: Subtotal before tax/discount
- tax: Tax amount
- discount: Discount amount
- total_amount: Final total amount

IMPORTANT: Extract EVERY line item you can see. Calculate total_amount
as the sum of all item amounts if not explicitly shown.

Respond with ONLY a valid JSON object matching these fields.
Do NOT include any other text outside the JSON."""


class ItemizedBillAgent:
    """Extracts itemized bill data from assigned pages."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract(self, pages: list[PageImage]) -> ItemizedBillInfo:
        if not pages:
            return ItemizedBillInfo()

        final_data = {"items": []}
        for page in pages:
            content = [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page.base64_image}",
                    },
                }
            ]
            message = HumanMessage(content=content)
            response = self.llm.invoke([message])
            data = safe_parse_json(response.content)
            
            for k, v in data.items():
                if k == "items" and isinstance(v, list):
                    final_data["items"].extend(v)
                elif v and (k not in final_data or final_data.get(k) is None):
                    final_data[k] = v
                # Prioritize financial totals from the LAST page (they usually span till the end)
                elif v and k in ['subtotal', 'tax', 'discount', 'total_amount'] and isinstance(v, (int, float)):
                    final_data[k] = v

        return ItemizedBillInfo(**final_data)
