import time
import structlog
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from app.config import get_settings
from app.workflow.graph import pipeline

logger = structlog.get_logger()

app = FastAPI(
    title="Claim Processing Pipeline",
    description="PDF claim processing with LangGraph multi-agent extraction",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "claim-processing-pipeline"}


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF file to process"),
):
    """
    Process a PDF claim document.

    1. Splits PDF into page images
    2. AI segregator classifies each page (NVIDIA NIM)
    3. Extraction agents process relevant pages
    4. Returns combined JSON with all extracted data
    """
    settings = get_settings()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted",
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB "
                   f"(max {settings.max_file_size_mb}MB)",
        )

    logger.info(
        "processing_claim",
        claim_id=claim_id,
        file_name=file.filename,
        file_size_mb=round(size_mb, 2),
    )

    try:
        # Run the LangGraph pipeline
        initial_state = {
            "claim_id": claim_id,
            "pdf_bytes": file_bytes,
            "start_time": time.time(),
            "page_images": [],
            "segregation_result": {},
            "id_result": None,
            "discharge_result": None,
            "bill_result": None,
            "final_result": None,
        }

        result = pipeline.invoke(initial_state)

        logger.info(
            "claim_processed",
            claim_id=claim_id,
            processing_time=result["final_result"]["processing_time_seconds"],
        )

        return result["final_result"]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "processing_failed",
            claim_id=claim_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
        )