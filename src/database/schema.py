from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class GeoLocation(BaseModel):
    latitude: Optional[float] = Field(None, description="Latitude degree (-90 to 90)")
    longitude: Optional[float] = Field(None, description="Longitude degree (-180 to 180)")
    region_name: Optional[str] = Field(None, description="Ecoregion or geographical area (e.g. Sahel, Indo-Gangetic Plain, Mediterranean)")

class EnvironmentalContext(BaseModel):
    # Soil Health
    soil_ph: Optional[float] = Field(None, description="Soil pH value (0-14)")
    soil_organic_carbon_pct: Optional[float] = Field(None, description="Soil Organic Carbon percentage (SOC %)")
    soil_moisture: Optional[str] = Field(None, description="Soil moisture level: low, moderate, high")
    microbial_activity: Optional[str] = Field(None, description="Microbial activity level: depleted, moderate, high")
    
    # Land Cover / Use
    land_use_type: Optional[str] = Field(None, description="Land use: monoculture, agroforestry, pastoral, degraded, forest")
    monoculture_pct: Optional[float] = Field(None, description="Percentage of land under single monoculture crop (0-100%)")
    habitat_fragmentation: Optional[str] = Field(None, description="Habitat fragmentation level: low, moderate, high, severe")
    crop_type: Optional[str] = Field(None, description="Primary crop cultivated (e.g. wheat, corn, soy, cotton)")
    
    # Biodiversity Indicators
    species_richness_index: Optional[float] = Field(None, description="Biodiversity species richness index score (0-100)")
    pollinator_index: Optional[str] = Field(None, description="Pollinator index: low, moderate, high")
    native_canopy_cover_pct: Optional[float] = Field(None, description="Percentage of native vegetative canopy cover (0-100%)")
    
    # Climate Factors
    rainfall_pattern: Optional[str] = Field(None, description="Rainfall pattern: low, erratic, moderate, high")
    aridity_index: Optional[str] = Field(None, description="Aridity level: hyper-arid, semi-arid, sub-humid, humid")
    avg_temperature_c: Optional[float] = Field(None, description="Average annual temperature in Celsius")
    
    # Human Impact
    pesticide_load: Optional[str] = Field(None, description="Pesticide load level: none, low, moderate, high, intensive")
    pollution_index: Optional[str] = Field(None, description="Chemical or industrial pollution index: low, moderate, high")
    
    # Spatial
    location: Optional[GeoLocation] = Field(None, description="Spatial coordinates or region name")

    def count_provided_metrics(self) -> int:
        """Counts how many key environmental variables are provided."""
        count = 0
        metrics = [
            self.soil_ph, self.soil_organic_carbon_pct, self.soil_moisture,
            self.land_use_type, self.monoculture_pct, self.crop_type,
            self.habitat_fragmentation, self.pollinator_index,
            self.rainfall_pattern, self.aridity_index, self.pesticide_load
        ]
        for m in metrics:
            if m is not None and str(m).strip() != "":
                count += 1
        return count

class QuantitativeEstimate(BaseModel):
    metric_name: str = Field(..., description="Target environmental metric to improve (e.g., Soil Organic Carbon)")
    baseline_value: str = Field(..., description="Estimated baseline value or condition")
    expected_change: str = Field(..., description="Expected quantitative percentage or absolute change (e.g., +15-25%)")
    timeframe: str = Field(..., description="Timeframe to achieve impact (e.g. 2-3 years)")

class ScientificEvidence(BaseModel):
    citation_source: str = Field(..., description="Credible source organization or journal (e.g. FAO, IPCC, IPBES, Nature)")
    study_title_or_report: str = Field(..., description="Report title or key scientific document")
    ecological_mechanism: str = Field(..., description="Specific biological, chemical, or ecological mechanism driving the result")

class StructuredRecommendation(BaseModel):
    is_clarification_needed: bool = Field(False, description="Flag indicating if mandatory metrics are missing")
    actionable_recommendation: str = Field(..., description="Specific, non-obvious, actionable intervention title & steps")
    scientific_reasoning: str = Field(..., description="In-depth ecological mechanism explaining why this intervention works")
    interwoven_variables_evaluated: List[str] = Field(..., description="List of at least 3 environmental variables cross-analyzed together")
    quantitative_estimates: List[QuantitativeEstimate] = Field(..., description="Quantitative metric improvement projections")
    time_horizon: str = Field(..., description="Implementation time horizon: Short, Medium, or Long term")
    citations: List[ScientificEvidence] = Field(..., description="Peer-reviewed studies, FAO, IPCC, or IPBES reference sources")
    confidence_level: float = Field(..., description="System confidence score between 0.0 and 1.0")

class ClarificationRequest(BaseModel):
    is_clarification_needed: bool = Field(True, description="Always true for clarification responses")
    current_detected_metrics: Dict[str, Any] = Field(..., description="Metrics detected from input so far")
    missing_metrics: List[str] = Field(..., description="Key missing environmental metrics")
    clarifying_question: str = Field(..., description="Polite, scientific follow-up question requesting specific metric details")
    suggested_inputs: List[str] = Field(..., description="List of metric fields user should provide")

class QueryAnalysisResult(BaseModel):
    parsed_context: EnvironmentalContext
    extracted_query_intent: str
