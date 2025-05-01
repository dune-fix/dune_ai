import asyncio
import argparse
import logging
from typing import Dict, Any

from config.logging_config import configure_logging
from dune_ai.services.twitter_sentinel import TwitterSentinel
from dune_ai.services.spice_trend_engine import SpiceTrendEngine
from dune_ai.services.sandworm_scanner import SandwormScanner
from dune_ai.blockchain.solana_client import SolanaClient


async def run_services(config: Dict[str, Any]):
    """
    Initialize and run all DUNE AI services concurrently
    """
    # Configure logging
    loggers = configure_logging()
    main_logger = logging.getLogger("main")
    main_logger.info("Starting DUNE AI services...")

    # Initialize Solana client
    solana_client = SolanaClient()
    await solana_client.initialize()

    # Initialize services
    twitter_sentinel = TwitterSentinel(solana_client)
    spice_trend_engine = SpiceTrendEngine(solana_client)
    sandworm_scanner = SandwormScanner(solana_client)

    # Start services as concurrent tasks
    tasks = [
        asyncio.create_task(twitter_sentinel.start_monitoring()),
        asyncio.create_task(spice_trend_engine.start_analysis()),
        asyncio.create_task(sandworm_scanner.start_scanning()),
    ]

    main_logger.info("All services started successfully")

    # Wait for all tasks to complete (or handle cancellation)
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        main_logger.info("Shutting down services...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        main_logger.info("All services stopped")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DUNE AI - Solana Meme Coin Analytics")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--monitor-only", action="store_true", help="Run only monitoring services without analysis")
    return parser.parse_args()


def main():
    """Main entry point for the application"""
    args = parse_arguments()

    # Load configuration
    config = {}  # In a real implementation, would load from args.config if specified

    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the services
    try:
        asyncio.run(run_services(config))
    except KeyboardInterrupt:
        print("\nShutting down DUNE AI...")
    except Exception as e:
        logging.getLogger("main").error(f"Critical error occurred: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())