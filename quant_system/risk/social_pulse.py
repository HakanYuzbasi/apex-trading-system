from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
import google.generativeai as genai

logger = logging.getLogger(__name__)

class SocialPulse:
    """
    Simulated social sentiment monitor (X/Reddit) integrated with an LLM.
    Detects retail 'pump' momentum which signals crowded trades (mean-reversion danger).
    """

    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        else:
            self.model = None
            logger.warning("SocialPulse initialized without GEMINI API key. Will use mock heuristic.")
            
        self.buzz_scores: dict[str, float] = {}
        self.buzz_history: dict[str, list[float]] = {}
        
    def get_retail_buzz_zscore(self, instrument_id: str) -> float:
        """
        Returns the existing buzz z-score for the given instrument without triggering an API call.
        """
        return self.buzz_scores.get(instrument_id, 0.0)
        
    def refresh_buzz(self, instrument_id: str) -> float:
        """
        Calculates a pseudo Z-Score for 'retail buzz' on an instrument.
        """
        # In a production system, this would stream X/Reddit comments.
        # We will use the LLM to 'score' a simulated daily digest instead.
        simulated_headlines = self._mock_scrape_socials(instrument_id)
        
        try:
            if self.model is not None:
                prompt = (
                    f"Review the following social media posts about {instrument_id}:\n"
                    f"{simulated_headlines}\n\n"
                    "On a scale of 0 to 10 (where 10 is 'insane retail pump/meme-stock territory' and 0 is 'no buzz'), "
                    "score the momentum. Reply ONLY with the integer score."
                )
                response = self.model.generate_content(prompt)
                try:
                    score_val = float(response.text.strip())
                except ValueError:
                    score_val = 5.0
            else:
                score_val = float(random.randint(0, 10))
        except Exception as e:
            logger.error("SocialPulse LLM Error for %s: %s", instrument_id, e)
            score_val = 5.0
            
        # Update history
        history = self.buzz_history.setdefault(instrument_id, [])
        history.append(score_val)
        if len(history) > 30:
            history.pop(0)
            
        # Calculate Z-Score
        if len(history) < 5:
            z_score = 0.0
        else:
            mean = sum(history) / len(history)
            variance = sum((x - mean) ** 2 for x in history) / len(history)
            std = variance ** 0.5
            
            if std == 0:
                z_score = 0.0
            else:
                z_score = (score_val - mean) / std
        
        self.buzz_scores[instrument_id] = z_score
        return z_score
        
    def _mock_scrape_socials(self, instrument_id: str) -> str:
        """Mocks retrieving posts from Twitter/Reddit."""
        # 10% chance of a mock 'retail pump'
        if random.random() < 0.10:
            return f"{instrument_id} to the MOON! 🚀🚀🚀 Apes together strong! Everyone is buying calls, squeeze incoming!"
        else:
            return f"Thinking of buying {instrument_id} here. Looks like a solid company long term. Decent fundamentals."
