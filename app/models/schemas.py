from pydantic import BaseModel, Field
from typing import Optional


class IdentityInfo(BaseModel):
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    id_type: Optional[str] = Field(
        None, description="e.g., Aadhaar, PAN, Passport"
    )
    id_number: Optional[str] = None
    policy_number: Optional[str] = None
    insurance_provider: Optional[str] = None
    contact_number: Optional[str] = None
    address: Optional[str] = None


class DischargeSummaryInfo(BaseModel):
    patient_name: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    diagnosis: Optional[list[str]] = None
    procedures_performed: Optional[list[str]] = None
    treating_physician: Optional[str] = None
    hospital_name: Optional[str] = None
    department: Optional[str] = None
    follow_up_instructions: Optional[str] = None


class BillLineItem(BaseModel):
    description: str
    quantity: Optional[float] = 1
    unit_price: Optional[float] = None
    amount: float


class ItemizedBillInfo(BaseModel):
    hospital_name: Optional[str] = None
    bill_number: Optional[str] = None
    bill_date: Optional[str] = None
    patient_name: Optional[str] = None
    items: list[BillLineItem] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total_amount: Optional[float] = None


class ProcessingResult(BaseModel):
    claim_id: str
    status: str = "success"
    segregation: dict = {}
    identity_info: Optional[IdentityInfo] = None
    discharge_summary: Optional[DischargeSummaryInfo] = None
    itemized_bill: Optional[ItemizedBillInfo] = None
    processing_time_seconds: Optional[float] = None
