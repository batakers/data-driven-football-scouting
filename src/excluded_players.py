"""
Excluded Players Registry
==========================
Players listed here are filtered out during data_cleaning.py so they do not
appear in predictions, similarity engine, or the dashboard.

Raw data is never modified - exclusions happen at the pipeline level only.
Temporal backtest (which reads raw data directly) is unaffected.

To add a player: append their name (exact match from Transfermarkt) to the
EXCLUDED_PLAYERS list below.
"""

# Players excluded from the active scouting pipeline.
# Reasons: retired, long-term banned, deceased, or otherwise no longer active.
EXCLUDED_PLAYERS: list[str] = [
    "Toni Kroos",       # Retired July 2024
]
