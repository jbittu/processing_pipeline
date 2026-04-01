from langchain_core.messages import HumanMessage
from app.services.llm_service import get_llm
from app.services.pdf_service import PageImage
from app.models.schemas import IdentityInfo
from app.utils.helpers import safe_parse_json
from tenacity import retry, stop_after_attempt, wait_exponential

EXTRACTION_PROMPT = """You are a medical insurance document data extractor.
You are given page(s) from identity documents (Aadhaar, PAN, passport,
insurance policy cards, etc.).

Extract ALL of the following fields. If a field is not found, set it to null.

Required fields:
- patient_name: Full name of the patient/policyholder
- date_of_birth: DOB in YYYY-MM-DD format if possible
- gender: Male/Female/Other
- id_type: Type of ID document (Aadhaar, PAN, Passport, etc.)
- id_number: The ID/document number
- policy_number: Insurance policy number
- insurance_provider: Name of insurance company
- contact_number: Phone number
- address: Full address

Respond with ONLY a valid JSON object matching these fields.
Do NOT include any other text outside the JSON."""


class IDAgent:
    """Extracts identity information from assigned pages."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract(self, pages: list[PageImage]) -> IdentityInfo:
        if not pages:
            return IdentityInfo()

        final_data = {}
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
                if v and not final_data.get(k):
                    final_data[k] = v

        return IdentityInfo(**final_data)
