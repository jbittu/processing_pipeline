from fastapi import FastAPI

app = FastAPI(
    title="Claim Processing Pipeline",
    description="PDF claim processing with LangGraph multi-agent extraction",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}