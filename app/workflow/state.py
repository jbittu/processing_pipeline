from typing import TypedDict, Optional, Any


class PipelineState(TypedDict):
    # Inputs
    claim_id: str
    pdf_bytes: bytes
    start_time: float

    # After PDF ingestion
    page_images: list          # list of PageImage objects

    # After segregation
    segregation_result: dict   # classifications + routing map

    # After extraction agents
    id_result: Optional[dict]
    discharge_result: Optional[dict]
    bill_result: Optional[dict]

    # Final output
    final_result: Optional[dict]
