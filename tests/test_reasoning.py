import pytest
from src.database.schema import EnvironmentalContext, StructuredRecommendation, ClarificationRequest
from src.engine.reasoning import EnvironmentalReasoningEngine
from src.engine.memory import ConversationMemory

def test_incomplete_input_triggers_clarification():
    engine = EnvironmentalReasoningEngine()
    incomplete_ctx = EnvironmentalContext(soil_organic_carbon_pct=0.3)  # Only 1 metric
    
    res = engine.evaluate(
        user_query="My soil is degrading",
        context=incomplete_ctx
    )
    
    assert isinstance(res, ClarificationRequest)
    assert res.is_clarification_needed is True
    assert len(res.suggested_inputs) > 0

def test_complete_input_produces_structured_recommendation():
    engine = EnvironmentalReasoningEngine()
    complete_ctx = EnvironmentalContext(
        soil_organic_carbon_pct=0.3,
        rainfall_pattern="low",
        aridity_index="semi-arid",
        crop_type="wheat",
        land_use_type="monoculture"
    )
    
    res = engine.evaluate(
        user_query="How to restore biodiversity?",
        context=complete_ctx
    )
    
    assert isinstance(res, StructuredRecommendation)
    assert res.is_clarification_needed is False
    assert len(res.interwoven_variables_evaluated) >= 3
    assert len(res.quantitative_estimates) > 0
    assert len(res.citations) > 0
    assert "FAO" in res.citations[0].citation_source or "IPCC" in res.citations[0].citation_source or "IPBES" in res.citations[0].citation_source

def test_conversation_memory_metric_extraction():
    mem = ConversationMemory()
    mem.add_user_message("My SOC is 0.4% and rainfall is low in a semi-arid wheat monoculture farm.")
    
    ctx = mem.get_accumulated_context()
    assert ctx.soil_organic_carbon_pct == 0.4
    assert ctx.rainfall_pattern == "low"
    assert ctx.crop_type == "wheat"
    assert ctx.land_use_type == "monoculture"
    assert ctx.count_provided_metrics() >= 3
