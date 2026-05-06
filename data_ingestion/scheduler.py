"""APScheduler-based automated data collection for the trading platform."""

from __future__ import annotations

import logging
import signal
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from data_storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_scheduler = None


def _ingest_cycle():
    """Run one full ingest -> NLP cycle."""
    from main import step_ingest, step_nlp

    db = DatabaseManager()
    try:
        step_ingest(db)
        step_nlp(db)
        logger.info("Scheduled pipeline cycle completed")
    except Exception:
        logger.exception("Scheduled pipeline cycle failed")


def start_scheduler(db: DatabaseManager | None = None):
    """Start the background scheduler for periodic data collection."""
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.warning("Scheduler already running")
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _ingest_cycle,
        trigger=IntervalTrigger(minutes=config.COLLECTION_INTERVAL_MINUTES),
        id="pipeline_cycle",
        name="Full Pipeline Cycle (Ingest + NLP)",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Scheduler started - pipeline runs every %d min", config.COLLECTION_INTERVAL_MINUTES)
    return _scheduler


def stop_scheduler():
    """Stop the background scheduler if running."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def run_once(db: DatabaseManager | None = None):
    """Manually trigger one pipeline cycle immediately."""
    db = db or DatabaseManager()
    _ingest_cycle()


def main():
    """CLI entry point: start scheduler and run until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    scheduler = start_scheduler()

    def _shutdown(signum, frame):
        logger.info("Received shutdown signal...")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Data scheduler running. Press Ctrl+C to stop.")
    logger.info("Running initial pipeline cycle...")
    run_once()

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        stop_scheduler()


if __name__ == "__main__":
    main()
