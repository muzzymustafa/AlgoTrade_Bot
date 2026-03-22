# utils/database.py
"""SQLite veritabanı — trade geçmişi, backtest sonuçları, konfigürasyon kaydı."""
import os
import sqlite3
import json
from datetime import datetime

import pandas as pd


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "algotrade.db")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection):
    """Tabloları oluştur (yoksa)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dt TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            size REAL,
            pnl REAL,
            commission REAL DEFAULT 0,
            bars_held INTEGER,
            regime INTEGER,
            exit_reason TEXT,
            mode TEXT DEFAULT 'backtest',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dt TEXT DEFAULT (datetime('now')),
            symbol TEXT,
            mode TEXT,
            params TEXT,
            start_cash REAL,
            end_value REAL,
            net_pl REAL,
            sharpe REAL,
            max_dd REAL,
            sqn REAL,
            total_trades INTEGER,
            win_rate REAL,
            source TEXT DEFAULT 'binance'
        );

        CREATE TABLE IF NOT EXISTS walk_forward_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dt TEXT DEFAULT (datetime('now')),
            window INTEGER,
            test_start TEXT,
            test_end TEXT,
            params TEXT,
            net_pl REAL,
            sharpe REAL,
            max_dd REAL,
            total_trades INTEGER,
            win_rate REAL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_dt ON trades(dt);
        CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_runs(symbol);
    """)
    conn.commit()


def save_trade(conn: sqlite3.Connection, trade: dict):
    """Tek bir trade kaydet."""
    conn.execute("""
        INSERT INTO trades (dt, symbol, side, entry_price, exit_price, size,
                           pnl, commission, bars_held, regime, exit_reason, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade.get("dt", datetime.now().isoformat()),
        trade.get("symbol", ""),
        trade.get("side", ""),
        trade.get("entry_price", 0),
        trade.get("exit_price", 0),
        trade.get("size", 0),
        trade.get("pnl", 0),
        trade.get("commission", 0),
        trade.get("bars_held"),
        trade.get("regime"),
        trade.get("exit_reason", ""),
        trade.get("mode", "backtest"),
    ))
    conn.commit()


def save_trades_bulk(conn: sqlite3.Connection, trades: list[dict]):
    """Birden fazla trade kaydet."""
    for t in trades:
        save_trade(conn, t)


def save_backtest_run(conn: sqlite3.Connection, metrics: dict):
    """Backtest sonucunu kaydet."""
    conn.execute("""
        INSERT INTO backtest_runs (symbol, mode, params, start_cash, end_value,
                                   net_pl, sharpe, max_dd, sqn, total_trades, win_rate, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        metrics.get("symbol", ""),
        metrics.get("mode", "backtest"),
        json.dumps(metrics.get("params", {})),
        metrics.get("start_cash", 10000),
        metrics.get("end_value", 0),
        metrics.get("net_pl", 0),
        metrics.get("sharpe"),
        metrics.get("max_dd"),
        metrics.get("sqn"),
        metrics.get("total_trades", 0),
        metrics.get("win_rate", 0),
        metrics.get("source", "binance"),
    ))
    conn.commit()


def get_trades(conn: sqlite3.Connection, symbol: str | None = None,
               limit: int = 100) -> pd.DataFrame:
    """Trade geçmişini DataFrame olarak döndür."""
    query = "SELECT * FROM trades"
    params = []
    if symbol:
        query += " WHERE symbol = ?"
        params.append(symbol)
    query += " ORDER BY dt DESC LIMIT ?"
    params.append(limit)
    return pd.read_sql_query(query, conn, params=params)


def get_backtest_runs(conn: sqlite3.Connection, limit: int = 50) -> pd.DataFrame:
    """Backtest geçmişini döndür."""
    return pd.read_sql_query(
        "SELECT * FROM backtest_runs ORDER BY dt DESC LIMIT ?",
        conn, params=(limit,),
    )


def get_stats(conn: sqlite3.Connection, symbol: str | None = None) -> dict:
    """Genel istatistikler."""
    where = f"WHERE symbol = '{symbol}'" if symbol else ""
    row = conn.execute(f"""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            MIN(pnl) as worst_trade,
            MAX(pnl) as best_trade
        FROM trades {where}
    """).fetchone()

    return {
        "total_trades": row[0] or 0,
        "wins": row[1] or 0,
        "win_rate": (row[1] / row[0]) if row[0] else 0,
        "total_pnl": row[2] or 0,
        "avg_pnl": row[3] or 0,
        "worst_trade": row[4] or 0,
        "best_trade": row[5] or 0,
    }
