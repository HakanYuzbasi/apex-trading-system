from __future__ import annotations

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
import google.generativeai as genai

logger = logging.getLogger("sentiment_warden")

class SentimentWarden:
    """
    AI-driven news monitor that uses Gemini 3 Flash to scan for structural landmines.
    Issues 24-hour Hard Vetoes on symbols with 'Category B' (Structural Change) news.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        gemini_api_key: Optional[str] = None,
        state_path: str = "run_state/v3/sentiment_vetoes.json",
    ) -> None:
        self._news_client = NewsClient(api_key, secret_key)
        self._gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._state_path = Path(state_path)
        self._vetoes: Dict[str, Dict[str, Any]] = {}

        if self._gemini_api_key:
            genai.configure(api_key=self._gemini_api_key)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            logger.warning("SentimentWarden: GEMINI_API_KEY not found. Sentiment analysis will be disabled.")
            self._model = None

        self._load_state()

    def _load_state(self) -> None:
        """Load veto state from disk."""
        if self._state_path.exists():
            try:
                with open(self._state_path, "r") as f:
                    self._vetoes = json.load(f)
                # Cleanup expired vetoes
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Failed to load sentiment vetoes: {e}")

    def _save_state(self) -> None:
        """Save veto state to disk."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(self._vetoes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sentiment vetoes: {e}")

    def _cleanup_expired(self) -> None:
        """Remove vetoes that have passed their 24h expiration."""
        now = datetime.now(timezone.utc).timestamp()
        expired = [s for s, v in self._vetoes.items() if v["expires_at"] < now]
        for s in expired:
            logger.info(f"SentimentWarden: Veto expired for {s}")
            del self._vetoes[s]
        if expired:
            self._save_state()

    def is_vetoed(self, symbol: str) -> bool:
        """Check if a symbol is currently under a structural veto."""
        self._cleanup_expired()
        return symbol in self._vetoes

    def get_veto_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the headline and expiration for a vetoed symbol."""
        return self._vetoes.get(symbol)

    async def scan_universe(self, symbols: List[str]) -> None:
        """Perform a full sentiment sweep for the provided universe."""
        if not self._model:
            return

        logger.info(f"SentimentWarden: Scanning news for {len(symbols)} symbols...")
        tasks = [self._analyze_symbol(s) for s in symbols]
        await asyncio.gather(*tasks)
        self._save_state()

    async def _analyze_symbol(self, symbol: str) -> None:
        """Fetch and analyze latest 5 headlines for a single symbol."""
        try:
            # 1. Fetch News
            request_params = NewsRequest(
                symbols=symbol,
                limit=5
            )
            news = await asyncio.to_thread(self._news_client.get_news, request_params)
            
            if not news.news:
                return

            headlines = [n.headline for n in news.news]
            combined_news = "\n".join([f"- {h}" for h in headlines])

            # 2. LLM Categorization
            prompt = f"""
Analyze the following latest headlines for {symbol}. Categorize the overall situation into:
- Category A: Market Noise / Standard Volatility (No structural change).
- Category B: Structural Change / Black Swan (CEO exit, Lawsuit, major Earnings Miss, Natural Disaster, Fraud, Regulatory Halt).

Output ONLY the category letter ('A' or 'B') followed by a colon and the single most critical headline reasoning.
If it is 'B', explain why.

Headlines:
{combined_news}
"""
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            result = response.text.strip()

            if result.startswith("B"):
                reason = result.split(":", 1)[1].strip() if ":" in result else result
                logger.warning(f"🚨 SENTIMENT VETO: {symbol} categorized as Category B. Reason: {reason}")
                
                # 24-hour Veto
                expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()
                self._vetoes[symbol] = {
                    "expires_at": expires_at,
                    "headline": reason,
                    "detected_at": datetime.now(timezone.utc).isoformat()
                }
            else:
                # If it was vetoed and now it's 'A', we might keep the veto for the full 24h as per requirement
                # "24-hour Hard Veto". We don't clear it here.
                pass

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    warden = SentimentWarden(
        api_key="...", secret_key="..." # Placeholder
    )
    # asyncio.run(warden.scan_universe(["AAPL", "TSLA"]))
