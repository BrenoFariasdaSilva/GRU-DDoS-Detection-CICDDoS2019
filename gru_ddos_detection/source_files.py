"""Locate direct day-level CICDDoS2019 source CSVs, excluding nested artifacts."""

from pathlib import Path
from typing import List


def find_source_csvs(data_dir: Path, source_day: str) -> List[Path]:
    """Return sorted raw CSV paths for 01-12, 03-11, or both days."""
    days = ("01-12", "03-11") if source_day.lower() == "both" else (source_day,)
    files = sorted(
        path for day in days for path in (data_dir / day).glob("*.csv") if path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No CSVs found for source day {source_day!r} under {data_dir}")
    return files
