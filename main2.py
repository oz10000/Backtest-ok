import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import time
import random
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =============================
# CONFIGURACIÓN
# =============================
DB_PATH = os.getenv("DB_PATH", "./data/backtest.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

INTERVAL = "1m"
START_DATE = "2026-01-01"
END_DATE = "2026-03-01"

MAX_WORKERS = 5
BATCH_SIZE = 20  # resultados por batch para DB

CAPITALS = [2, 5, 10, 25, 50, 100]
LEVERAGES = [2,3,4,5,10,20,50,100]

ADX_PERIODS = [3,7,10,14,17,20]
ATR_PERIODS = [3,7,10,14,17,20]

TRAILING_ENABLED = True
BREAK_EVEN_ENABLED = True
COMMISSION_RATE = 0.0004  # 0.04% por trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =============================
# SQLITE
# =============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT,
            time INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY(symbol, time)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            symbol TEXT,
            timestamp TEXT,
            adx_period INTEGER,
            atr_period INTEGER,
            sl_mult REAL,
            tp_mult REAL,
            capital REAL,
            leverage REAL,
            win_rate REAL,
            roi REAL,
            trades INTEGER,
            viable INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS processed_symbols (
            symbol TEXT PRIMARY KEY
        )
    """)
    conn.commit()
    conn.close()

def mark_symbol_processed(symbol):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO processed_symbols (symbol) VALUES (?)", (symbol,))
    conn.commit()
    conn.close()

def is_symbol_processed(symbol):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM processed_symbols WHERE symbol=?", (symbol,))
    res = cur.fetchone()
    conn.close()
    return res is not None

def save_klines_to_db(symbol, df):
    conn = sqlite3.connect(DB_PATH)
    data = [(symbol, row['time'], row['open'], row['high'], row['low'], row['close'], row['volume'])
            for _, row in df.iterrows()]
    conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", data)
    conn.commit()
    conn.close()

def load_klines_from_db(symbol, start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    start_ts = int(pd.to_datetime(start_date).timestamp()*1000)
    end_ts = int(pd.to_datetime(end_date).timestamp()*1000)
    df = pd.read_sql("SELECT * FROM ohlcv WHERE symbol=? AND time BETWEEN ? AND ? ORDER BY time ASC",
                     conn, params=(symbol, start_ts, end_ts))
    conn.close()
    return df if not df.empty else None

def save_results_batch(results_list):
    if not results_list: return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("""
        INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, results_list)
    conn.commit()
    conn.close()

# =============================
# BINANCE UTILITIES
# =============================
def get_all_futures_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['contractType']=='PERPETUAL']
        logging.info(f"{len(symbols)} símbolos de futuros encontrados")
        return symbols
    except Exception as e:
        logging.error(f"Error obteniendo símbolos: {e}")
        return []

def fetch_klines_range(symbol, start_str, end_str, interval='1m', limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)
    all_klines = []

    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "endTime": end_ts, "limit": limit}
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if not data: return pd.DataFrame()
                    all_klines.extend(data)
                    start_ts = data[-1][0] + 60_000
                    time.sleep(0.1)
                    break
                else:
                    wait = (2**attempt)+random.random()
                    logging.warning(f"{symbol} status {r.status_code}, retry en {wait:.2f}s")
                    time.sleep(wait)
            except Exception as e:
                wait = (2**attempt)+random.random()
                logging.error(f"{symbol} excepción {e}, retry en {wait:.2f}s")
                time.sleep(wait)
        else:
            logging.error(f"{symbol} fallo definitivo en fetch_klines")
            return pd.DataFrame()
    df = pd.DataFrame(all_klines, columns=['time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore'])
    df = df[['time','open','high','low','close','volume']]
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# =============================
# INDICADORES
# =============================
def compute_indicators(df):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    for period in ATR_PERIODS:
        df[f'atr_{period}'] = tr.rolling(period).mean()
    for period in ADX_PERIODS:
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up>down)&(up>0), up, 0)
        minus_dm = np.where((down>up)&(down>0), down, 0)
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period).mean()/tr.ewm(span=period).mean()
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period).mean()/tr.ewm(span=period).mean()
        dx = 100 * abs(plus_di - minus_di)/(plus_di + minus_di + 1e-10)
        df[f'adx_{period}'] = dx.ewm(span=period).mean()
        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di
    return df

def classify_asset(df):
    adx_mean = df[[f'adx_{p}' for p in ADX_PERIODS]].mean().mean()
    if adx_mean > 25: return 'continuacion'
    elif adx_mean < 20: return 'lateral'
    else: return 'rebote'

# =============================
# SEÑALES
# =============================
def get_entry_signal(df, i, side, adx_period):
    adx = df[f'adx_{adx_period}'].iloc[i]
    plus_di = df[f'plus_di_{adx_period}'].iloc[i]
    minus_di = df[f'minus_di_{adx_period}'].iloc[i]
    if side=='long':
        return adx>25 and plus_di>minus_di
    else:
        return adx>25 and minus_di>plus_di

# =============================
# BACKTEST DUAL
# =============================
def backtest_dual(df, adx_p, atr_p, sl_mult, tp_mult, capital, leverage):
    trades_long, trades_short = [], []
    capital_asig = capital*0.5
    pos_long, pos_short = None, None

    for i in range(max(adx_p, atr_p), len(df)):
        atr = df[f'atr_{atr_p}'].iloc[i]
        close, high, low = df['close'].iloc[i], df['high'].iloc[i], df['low'].iloc[i]

        # --- Long ---
        if pos_long:
            exit_price = None
            if low <= pos_long['sl']: exit_price = pos_long['sl']
            if high >= pos_long['tp']: exit_price = pos_long['tp']
            if TRAILING_ENABLED and pos_long.get('trailing_activated') and low <= pos_long.get('trailing_stop',0): exit_price = pos_long['trailing_stop']
            if BREAK_EVEN_ENABLED and not pos_long.get('be_activated') and close - pos_long['entry_price'] >= atr:
                pos_long['sl'] = pos_long['entry_price']
                pos_long['be_activated'] = True
            if exit_price:
                size = (capital_asig*leverage)/pos_long['entry_price']
                pnl = (exit_price - pos_long['entry_price'])*size
                pnl -= 2*COMMISSION_RATE*size*pos_long['entry_price']
                trades_long.append(pnl)
                pos_long = None

        if pos_long is None and get_entry_signal(df, i, 'long', adx_p):
            entry_price = close
            pos_long = {'entry_price': entry_price, 'sl': entry_price - sl_mult*atr,
                        'tp': entry_price + tp_mult*atr, 'trailing_activated': False, 'be_activated': False}

        # --- Short ---
        if pos_short:
            exit_price = None
            if high >= pos_short['sl']: exit_price = pos_short['sl']
            if low <= pos_short['tp']: exit_price = pos_short['tp']
            if TRAILING_ENABLED and pos_short.get('trailing_activated') and high >= pos_short.get('trailing_stop',0): exit_price = pos_short['trailing_stop']
            if BREAK_EVEN_ENABLED and not pos_short.get('be_activated') and pos_short['entry_price'] - close >= atr:
                pos_short['sl'] = pos_short['entry_price']
                pos_short['be_activated'] = True
            if exit_price:
                size = (capital_asig*leverage)/pos_short['entry_price']
                pnl = (pos_short['entry_price'] - exit_price)*size
                pnl -= 2*COMMISSION_RATE*size*pos_short['entry_price']
                trades_short.append(pnl)
                pos_short = None

        if pos_short is None and get_entry_signal(df, i, 'short', adx_p):
            entry_price = close
            pos_short = {'entry_price': entry_price, 'sl': entry_price + sl_mult*atr,
                         'tp': entry_price - tp_mult*atr, 'trailing_activated': False, 'be_activated': False}

    total_trades = len(trades_long) + len(trades_short)
    roi = (sum(trades_long)+sum(trades_short))/capital if total_trades>0 else 0
    win_rate = sum([1 for x in trades_long+trades_short if x>0])/total_trades if total_trades>0 else 0
    viable = 1 if win_rate>0.5 else 0
    return win_rate, roi, total_trades, viable

# =============================
# PROCESAMIENTO POR SÍMBOLO CON PROGRESO
# =============================
def process_symbol(symbol, pbar_symbol):
    try:
        if is_symbol_processed(symbol):
            tqdm.write(f"{symbol} ya procesado, saltando")
            pbar_symbol.update(1)
            return

        df = load_klines_from_db(symbol, START_DATE, END_DATE)
        if df is None or len(df)<100:
            df = fetch_klines_range(symbol, START_DATE, END_DATE, INTERVAL)
            if df is not None and not df.empty:
                save_klines_to_db(symbol, df)
        if df is None or df.empty:
            tqdm.write(f"{symbol} sin datos")
            pbar_symbol.update(1)
            return

        df = compute_indicators(df)
        behavior = classify_asset(df)

        results_batch = []
        total_combinations = len(ADX_PERIODS)*len(ATR_PERIODS)*3*3*len(CAPITALS)*len(LEVERAGES)
        with tqdm(total=total_combinations, desc=f"{symbol} batches", leave=False) as pbar_batch:
            for adx_p in ADX_PERIODS:
                for atr_p in ATR_PERIODS:
                    sl_mults = [1.0,1.5,2.0] if behavior=='rebote' else [1.5,2.0,2.5]
                    tp_mults = [2.0,3.0,4.0] if behavior=='continuacion' else [3.0,4.0,5.0]
                    for sl in sl_mults:
                        for tp in tp_mults:
                            for capital in CAPITALS:
                                for leverage in LEVERAGES:
                                    win_rate, roi, trades_count, viable = backtest_dual(df, adx_p, atr_p, sl, tp, capital, leverage)
                                    results_batch.append((symbol, datetime.now(), adx_p, atr_p, sl, tp, capital, leverage,
                                                          win_rate, roi, trades_count, viable))
                                    if len(results_batch)>=BATCH_SIZE:
                                        save_results_batch(results_batch)
                                        results_batch.clear()
                                    pbar_batch.update(1)

        if results_batch:
            save_results_batch(results_batch)

        mark_symbol_processed(symbol)
        tqdm.write(f"{symbol} procesado ({behavior})")
        pbar_symbol.update(1)

    except Exception as e:
        logging.error(f"{symbol} fallo en procesamiento: {e}")
        pbar_symbol.update(1)

# =============================
# MAIN CON BARRA DE PROGRESO GLOBAL
# =============================
def main():
    init_db()
    symbols = get_all_futures_symbols()
    if not symbols:
        logging.error("No se encontraron símbolos, abortando ejecución.")
        return
    with tqdm(total=len(symbols), desc="Total symbols") as pbar_symbols:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_symbol, symbol, pbar_symbols) for symbol in symbols]
            for _ in as_completed(futures):
                pass

if __name__=="__main__":
    main()
