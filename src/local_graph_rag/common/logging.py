"""Shared logging setup for CLI entry points."""

import logging

LOG_FORMAT = "%(levelname)s %(name)s — %(message)s"


def configure_cli_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

