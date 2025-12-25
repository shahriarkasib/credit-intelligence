"""Storage module for Credit Intelligence."""

from .mongodb import CreditIntelligenceDB, get_db

__all__ = ["CreditIntelligenceDB", "get_db"]
