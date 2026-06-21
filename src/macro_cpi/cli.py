"""Command-line entry point. Stages: ingest, features, validate, report, all."""
from __future__ import annotations

import argparse

from macro_cpi import config
from macro_cpi.db import get_engine, init_db
from macro_cpi.features.build import build_feature_panel
from macro_cpi.ingestion import fred, market, treasury
from macro_cpi.logging_conf import get_logger
from macro_cpi.models.walk_forward import walk_forward
from macro_cpi.reporting.report import write_report

logger = get_logger(__name__)


def cmd_ingest(_: argparse.Namespace) -> None:
    """Run all ingestion sources into the database."""
    engine = get_engine()
    init_db(engine)
    total = 0
    total += fred.run()
    total += market.run()
    total += treasury.run()
    logger.info("ingest complete: %d rows written", total)


def cmd_report(_: argparse.Namespace) -> None:
    """Build features, run walk-forward validation, and write the markdown report."""
    engine = get_engine()
    panel = build_feature_panel(engine, config.MODEL)
    result = walk_forward(panel, config.MODEL)
    path = write_report(result, panel, config.MODEL)
    logger.info("report written: %s", path)


def cmd_all(args: argparse.Namespace) -> None:
    """Run ingest then report end-to-end."""
    cmd_ingest(args)
    cmd_report(args)


def main() -> None:
    """Parse arguments and dispatch to the selected stage."""
    parser = argparse.ArgumentParser(prog="macro-cpi", description="Macro CPI forecasting engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="fetch and store all sources").set_defaults(func=cmd_ingest)
    sub.add_parser("report", help="build features, validate, write report").set_defaults(
        func=cmd_report
    )
    sub.add_parser("all", help="ingest then report").set_defaults(func=cmd_all)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
