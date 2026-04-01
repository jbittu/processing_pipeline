import time
from langgraph.graph import StateGraph, START, END
from app.workflow.state import PipelineState
from app.services.pdf_service import PDFService
from app.agents.segregator import SegregatorAgent
from app.agents.id_agent import IDAgent
from app.agents.discharge_agent import DischargeSummaryAgent
from app.agents.bill_agent import ItemizedBillAgent


# ── Node Functions ──────────────────────────────────────────

def ingest_pdf(state: PipelineState) -> dict:
    """Node: Convert PDF bytes into page images."""
    pdf_service = PDFService()
    pages = pdf_service.split_to_page_images(state["pdf_bytes"])
    return {"page_images": pages}


def segregate_pages(state: PipelineState) -> dict:
    """Node: Classify each page into document types."""
    agent = SegregatorAgent()
    result = agent.classify_all_pages(state["page_images"])
    return {"segregation_result": result}


def extract_identity(state: PipelineState) -> dict:
    """Node: Extract identity info from relevant pages."""
    routing = state["segregation_result"].get("routing", {})
    page_numbers = routing.get("identity_document", [])

    pdf_service = PDFService()
    pages = pdf_service.extract_pages_subset(
        state["page_images"], page_numbers
    )

    agent = IDAgent()
    result = agent.extract(pages)
    return {"id_result": result.model_dump()}


def extract_discharge_summary(state: PipelineState) -> dict:
    """Node: Extract discharge summary from relevant pages."""
    routing = state["segregation_result"].get("routing", {})
    page_numbers = routing.get("discharge_summary", [])

    pdf_service = PDFService()
    pages = pdf_service.extract_pages_subset(
        state["page_images"], page_numbers
    )

    agent = DischargeSummaryAgent()
    result = agent.extract(pages)
    return {"discharge_result": result.model_dump()}


def extract_itemized_bill(state: PipelineState) -> dict:
    """Node: Extract itemized bill data from relevant pages."""
    routing = state["segregation_result"].get("routing", {})
    page_numbers = routing.get("itemized_bill", [])

    pdf_service = PDFService()
    pages = pdf_service.extract_pages_subset(
        state["page_images"], page_numbers
    )

    agent = ItemizedBillAgent()
    result = agent.extract(pages)
    return {"bill_result": result.model_dump()}


def aggregate_results(state: PipelineState) -> dict:
    """Node: Combine all extraction results into final output."""
    final = {
        "claim_id": state["claim_id"],
        "status": "success",
        "segregation": {
            "page_classifications": state["segregation_result"].get(
                "classifications", []
            ),
            "routing_summary": state["segregation_result"].get(
                "routing", {}
            ),
        },
        "extracted_data": {
            "identity_info": state.get("id_result"),
            "discharge_summary": state.get("discharge_result"),
            "itemized_bill": state.get("bill_result"),
        },
        "processing_time_seconds": round(
            time.time() - state["start_time"], 2
        ),
    }
    return {"final_result": final}


# ── Build the Graph ─────────────────────────────────────────

def build_pipeline_graph() -> StateGraph:
    """
    Constructs the LangGraph workflow:

    START → ingest_pdf → segregate_pages
              → extract_identity ─────────┐
              → extract_discharge_summary ─┤→ aggregate → END
              → extract_itemized_bill ─────┘
    """

    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("ingest_pdf", ingest_pdf)
    graph.add_node("segregate_pages", segregate_pages)
    graph.add_node("extract_identity", extract_identity)
    graph.add_node("extract_discharge_summary", extract_discharge_summary)
    graph.add_node("extract_itemized_bill", extract_itemized_bill)
    graph.add_node("aggregate_results", aggregate_results)

    # Define edges (the DAG)
    graph.add_edge(START, "ingest_pdf")
    graph.add_edge("ingest_pdf", "segregate_pages")

    # After segregation, fan out to 3 extraction agents
    graph.add_edge("segregate_pages", "extract_identity")
    graph.add_edge("segregate_pages", "extract_discharge_summary")
    graph.add_edge("segregate_pages", "extract_itemized_bill")

    # All agents converge into aggregator
    graph.add_edge("extract_identity", "aggregate_results")
    graph.add_edge("extract_discharge_summary", "aggregate_results")
    graph.add_edge("extract_itemized_bill", "aggregate_results")

    graph.add_edge("aggregate_results", END)

    return graph.compile()


# Singleton compiled graph
pipeline = build_pipeline_graph()
