"""SQLite persistence via SQLAlchemy. One observations table, point-in-time aware."""
from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session

from macro_cpi import config


class Base(DeclarativeBase):
    """Declarative base."""


class Observation(Base):
    """A single point-in-time observation: a value for a series at a reference date."""

    __tablename__ = "observations"

    series_id = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)  # reference period end
    value = Column(Float, nullable=True)
    source = Column(String, nullable=False)
    release_date = Column(Date, nullable=False)  # when this value became publicly known
    fetched_at = Column(DateTime, nullable=False)

    __table_args__ = (UniqueConstraint("series_id", "date", name="uq_series_date"),)


def get_engine(db_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine, defaulting to the configured DB URL."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(db_url or config.DB_URL, future=True)


def init_db(engine: Engine) -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(engine)


def upsert_observations(engine: Engine, rows: list[dict]) -> int:
    """Insert-or-replace observations keyed on (series_id, date); return row count written."""
    if not rows:
        return 0
    init_db(engine)
    now = datetime.now(UTC)
    for r in rows:
        r.setdefault("fetched_at", now)
    stmt = sqlite_insert(Observation).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["series_id", "date"],
        set_={
            "value": stmt.excluded.value,
            "source": stmt.excluded.source,
            "release_date": stmt.excluded.release_date,
            "fetched_at": stmt.excluded.fetched_at,
        },
    )
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()
    return len(rows)


def load_series(engine: Engine, series_id: str) -> pd.DataFrame:
    """Load one series as a DataFrame with columns date, value, release_date, sorted by date."""
    with Session(engine) as session:
        stmt = (
            select(Observation.date, Observation.value, Observation.release_date)
            .where(Observation.series_id == series_id)
            .order_by(Observation.date)
        )
        records = session.execute(stmt).all()
    df = pd.DataFrame(records, columns=["date", "value", "release_date"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def load_all_series_ids(engine: Engine) -> list[str]:
    """Return the distinct series_ids currently stored."""
    with Session(engine) as session:
        rows = session.execute(select(Observation.series_id).distinct()).all()
    return sorted(r[0] for r in rows)


def release_date_for(reference_period_end: date, lag_days: int) -> date:
    """Compute the public release date as reference period end plus a publication lag."""
    return reference_period_end + pd.Timedelta(days=lag_days).to_pytimedelta()
