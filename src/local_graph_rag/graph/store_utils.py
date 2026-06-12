"""Shared graph-store utilities."""

import re


def slugify(name: str) -> str:
    """Return a normalised ASCII slug for an entity name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")

