"""Frames in raw_sql's return format without WRDS (draft A4): a frame goes through an in-memory SQLite table and
pd.read_sql_query on a SQLAlchemy connection with raw_sql's arguments (coerce_float=True, dtype_backend="numpy_nullable"),
as wrds.Connection.raw_sql does on its SQLAlchemy connection. pandas infers dtypes from the returned values, so an
all-NULL column comes back as string (pd.NA) and a 0-row result as object columns, whatever the SQL column type."""
import pandas as pd
import sqlalchemy as sa

RAW_SQL_KW = {"coerce_float": True, "dtype_backend": "numpy_nullable"}


def raw_frame(df, null_cols=(), rows=None):
    """df -> SQLite -> read back. null_cols are set to NULL in every row; rows=0 reads a 0-row result."""
    d = df.copy()
    d["time_m"] = [t.isoformat() if hasattr(t, "isoformat") else t for t in d["time_m"]]
    for c in null_cols:
        d[c] = None
    eng = sa.create_engine("sqlite://")
    try:
        with eng.connect() as con:
            d.to_sql("t", con, index=False)
            where = " where 1 = 0" if rows == 0 else ""
            return pd.read_sql_query(sa.text("select * from t" + where), con, **RAW_SQL_KW)
    finally:
        eng.dispose()


def typed_null(raw, typed, cols):
    """The reference input: raw with cols as all-NA of the nullable dtype the column has in typed (a read with values)."""
    d = raw.copy()
    for c in cols:
        d[c] = pd.array([pd.NA] * len(d), dtype=typed[c].dtype)
    return d
