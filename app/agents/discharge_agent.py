from langchain_core.messages import HumanMessage
from app.services.llm_service import get_llm
from app.services.pdf_service import PageImage
from app.models.schemas import DischargeSummaryInfo
from app.utils.helpers import safe_parse_json

EXTRACTION_PROMPT = """You are a medical document data extractor.
You are given page(s) from a hospital discharge summary.

Extract ALL of the following fields. If a field is not found, set it to null.

Required fields:
- patient_name: Full name of the patient
- admission_date: Date of admission (YYYY-MM-DD format if possible)
- discharge_date: Date of discharge (YYYY-MM-DD format if possible)
- diagnosis: List of diagnoses (as array of strings)
- procedures_performed: List of procedures (as array of strings)
- treating_physician: Name of the treating doctor
- hospital_name: Name of the hospital
- department: Department/ward name
- follow_up_instructions: Any follow-up advice given

Respond with ONLY a valid JSON object matching these fields.
Do NOT include any other text outside the JSON."""


class DischargeSummaryAgent:
    """Extracts discharge summary data from assigned pages."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

    def extract(self, pages: list[PageImage]) -> DischargeSummaryInfo:
        if not pages:
            return DischargeSummaryInfo()

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
                if isinstance(v, list):
                    final_data.setdefault(k, []).extend(v)
                elif v and not final_data.get(k):
                    final_data[k] = v

        # Deduplicate list elements just in case multiple pages repeat the same diagnosis
        for list_key in ['diagnosis', 'procedures_performed']:
            if list_key in final_data and isinstance(final_data[list_key], list):
                # Using dict.fromkeys to keep insertion order but remove duplicates
                final_data[list_key] = list(dict.fromkeys(final_data[list_key]))

        return DischargeSummaryInfo(**final_data)
