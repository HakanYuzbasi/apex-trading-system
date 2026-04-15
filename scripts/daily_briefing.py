import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import google.generativeai as genai

from quant_system.analytics.notifier import TelegramNotifier

logger = logging.getLogger("daily_briefing")

async def generate_daily_briefing() -> None:
    """
    Gathers daily metrics, feeds them to Gemini 3 as the 'Chief of Staff',
    and dispatches the generated summary to Telegram.
    """
    root_dir = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    
    # Read trace
    trace_path = root_dir / "quant_system" / "logs" / "decision_trace.json"
    trace_data = "No decision trace generated today."
    if trace_path.exists():
        with open(trace_path, "r", encoding="utf-8") as f:
            trace_data = f.read()[-2000:] # Last 2000 chars for context
            
    # Read adversarial
    adv_path = root_dir / "run_state" / "adversarial_results.json"
    adv_data = "No adversarial results found."
    if adv_path.exists():
        with open(adv_path, "r", encoding="utf-8") as f:
            adv_data = f.read()
            
    # TCA Mock (normally queried from DB)
    tca_data = "TCA Average Slippage: ~1.2bps. Execution Algo: OBI_Sniper active."

    prompt = (
        f"You are the Fund Manager's Chief of Staff. Today is {now.strftime('%Y-%m-%d')}.\n"
        "Review today's algorithmic trading performance logs and explain what happened concisely.\n"
        "We are running an ensemble strategy with both Kalman Pairs (mean-reversion) and BreakoutPod (volatility).\n"
        "Focus on highlighting the effectiveness of the system, any vetoes hit by the Sentiment Warden or Social Pulse, "
        "and overall drawdown controls or TCA metrics.\n\n"
        "### TELEMETRY DATA ###\n"
        f"DECISION TRACE:\n{trace_data}\n\n"
        f"ADVERSARIAL RESULTS:\n{adv_data}\n\n"
        f"TCA SUMMARY:\n{tca_data}\n\n"
        "Deep Reasoning: You must analyze why the bot succeeded or failed today. Did the OBISniper save us more than the Kalman signals generated? Did the Warden's Veto protect us from a specific sector event?\n"
        "Do not include pleasantries. Start directly with the briefing, making it sound very professional, suited for a Telegram message to a hedge fund manager."
    )
    
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("No GEMINI_API_KEY provided. Cannot generate briefing.")
        return
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        briefing_text = response.text
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        return

    # Persist to archive
    archive_path = root_dir / "run_state" / "v3" / "briefings_archive.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    
    archive_data = []
    if archive_path.exists():
        try:
            with open(archive_path, "r", encoding="utf-8") as f:
                archive_data = json.load(f)
        except json.JSONDecodeError:
            archive_data = []
            
    archive_data.append({
        "timestamp": now.isoformat(),
        "date": now.strftime('%Y-%m-%d'),
        "content": briefing_text.strip()
    })
    
    # Keep only the last 30 briefings
    if len(archive_data) > 30:
        archive_data = archive_data[-30:]
        
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(archive_data, f, indent=2)

    notifier = TelegramNotifier()
    msg = f"📊 *Chief of Staff EOD Briefing*\n\n{briefing_text.strip()}"
    await notifier.send_message(msg)
    logger.info("Daily briefing generated and sent to Telegram.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_daily_briefing())
