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


def _paper_trading_cycle():
    """Run one full paper trading cycle: ingest -> NLP -> RL inference -> execute."""
    from paper_trader import run_paper_trading_cycle

    try:
        run_paper_trading_cycle()
        logger.info("Scheduled paper trading cycle completed")
    except Exception:
        logger.exception("Scheduled paper trading cycle failed")


def start_scheduler(db: DatabaseManager | None = None, paper_trading: bool = False):
    """Start the background scheduler for periodic data collection.

    Args:
        db: DatabaseManager instance (created fresh if None).
        paper_trading: If True, runs full paper trading cycle (ingest+NLP+RL+execute).
                       If False, runs only ingest+NLP.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.warning("Scheduler already running")
        return _scheduler

    cycle_fn = _paper_trading_cycle if paper_trading else _ingest_cycle
    cycle_name = "Paper Trading (Ingest + NLP + RL + Execute)" if paper_trading else "Ingest + NLP"

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        cycle_fn,
        trigger=IntervalTrigger(minutes=config.COLLECTION_INTERVAL_MINUTES),
        id="pipeline_cycle",
        name=cycle_name,
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Scheduler started - %s runs every %d min", cycle_name, config.COLLECTION_INTERVAL_MINUTES)
    return _scheduler


def stop_scheduler():
    """Stop the background scheduler if running."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def run_once(db: DatabaseManager | None = None, paper_trading: bool = False):
    """Manually trigger one pipeline cycle immediately."""
    db = db or DatabaseManager()
    if paper_trading:
        _paper_trading_cycle()
    else:
        _ingest_cycle()


def main():
    """CLI entry point: start scheduler and run until interrupted."""
    import argparse
    parser = argparse.ArgumentParser(description="Data scheduler for NLP-RL Platform")
    parser.add_argument("--paper", action="store_true",
                        help="Run full paper trading cycles (ingest+NLP+RL+execute)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    scheduler = start_scheduler(paper_trading=args.paper)

    def _shutdown(signum, frame):
        logger.info("Received shutdown signal...")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    mode = "Paper Trading" if args.paper else "Data Pipeline"
    logger.info("%s scheduler running. Press Ctrl+C to stop.", mode)
    logger.info("Running initial cycle...")
    run_once(paper_trading=args.paper)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        stop_scheduler()


if __name__ == "__main__":
    main()
