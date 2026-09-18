import re
from typing import List, Dict, Any, Optional
from src.database.schema import EnvironmentalContext, GeoLocation

class ConversationMemory:
    """
    Stateful memory manager for Darukaa.Earth.
    Maintains chat history and dynamically aggregates detected environmental metrics
    across multi-turn conversations.
    """
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.accumulated_context: EnvironmentalContext = EnvironmentalContext()

    def add_user_message(self, text: str):
        """Adds user message and extracts environmental metrics from text."""
        self.history.append({"role": "user", "content": text})
        self._extract_and_update_metrics(text)

    def add_assistant_message(self, text: str):
        """Adds assistant message to history."""
        self.history.append({"role": "assistant", "content": text})

    def update_context(self, new_context: EnvironmentalContext):
        """Merges new environmental context parameters into accumulated state."""
        for field, value in new_context.model_dump().items():
            if value is not None:
                setattr(self.accumulated_context, field, value)

    def _extract_and_update_metrics(self, text: str):
        """Regex and heuristic extractor for common environmental metric mentions."""
        text_lower = text.lower()
        
        # Extract Soil Organic Carbon (SOC %)
        soc_match = re.search(r'(?:soc|soil organic carbon|carbon)\s*(?:is|=|:)?\s*([0-9.]+)\s*%', text_lower)
        if soc_match:
            try:
                self.accumulated_context.soil_organic_carbon_pct = float(soc_match.group(1))
            except ValueError:
                pass
                
        # Extract Soil pH
        ph_match = re.search(r'ph\s*(?:is|=|:)?\s*([0-9.]+)', text_lower)
        if ph_match:
            try:
                self.accumulated_context.soil_ph = float(ph_match.group(1))
            except ValueError:
                pass

        # Extract Rainfall
        if 'low rainfall' in text_lower or 'arid' in text_lower or 'dry' in text_lower or 'drought' in text_lower:
            self.accumulated_context.rainfall_pattern = "low"
            self.accumulated_context.aridity_index = "semi-arid"
        elif 'high rainfall' in text_lower or 'heavy rain' in text_lower or 'wet' in text_lower:
            self.accumulated_context.rainfall_pattern = "high"
            self.accumulated_context.aridity_index = "sub-humid"

        # Extract Crop / Monoculture
        if 'wheat' in text_lower:
            self.accumulated_context.crop_type = "wheat"
        elif 'corn' in text_lower or 'maize' in text_lower:
            self.accumulated_context.crop_type = "corn"
        elif 'soy' in text_lower:
            self.accumulated_context.crop_type = "soy"

        if 'monoculture' in text_lower or 'single crop' in text_lower:
            self.accumulated_context.land_use_type = "monoculture"
            if self.accumulated_context.monoculture_pct is None:
                self.accumulated_context.monoculture_pct = 85.0

        if 'agroforestry' in text_lower:
            self.accumulated_context.land_use_type = "agroforestry"

        # Extract Fragmentation & Pesticide
        if 'fragmented' in text_lower or 'fragmentation' in text_lower:
            self.accumulated_context.habitat_fragmentation = "high"
        if 'pesticide' in text_lower or 'chemical' in text_lower:
            self.accumulated_context.pesticide_load = "high"

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def get_accumulated_context(self) -> EnvironmentalContext:
        return self.accumulated_context

    def clear(self):
        self.history = []
        self.accumulated_context = EnvironmentalContext()
