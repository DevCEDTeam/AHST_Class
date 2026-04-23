"""Live Alpha Vantage data client.

Drop-in replacement for yfinance daily OHLCV pulls. Uses the
TIME_SERIES_DAILY_ADJUSTED endpoint (Premium) with fallback to
TIME_SERIES_DAILY. Includes a parquet-backed TTL cache and
exponential-backoff retry with throttling for plan rate limits.

    >>> from alpha_vantage_live import AlphaVantageClient, AlphaVantageConfig, CacheManager
    >>> cfg = AlphaVantageConfig(api_key='YOUR_KEY')
    >>> client = AlphaVantageClient(cfg, CacheManager(cfg))
    >>> gldm = client.get_daily('GLDM')
    >>> prices = client.download(['GLDM', 'TLT'])   # wide DataFrame of adjusted closes

Docs: https://www.alphavantage.co/documentation
Premium: https://www.alphavantage.co/premium/
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

# Load .env from the directory containing this file
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())


__all__ = [
    'AlphaVantageConfig',
    'CacheManager',
    'AlphaVantageClient',
    'download',
]


@dataclass
class AlphaVantageConfig:
    """Configuration for the Alpha Vantage client."""

    api_key: str = field(default_factory=lambda: os.environ.get('ALPHA_VANTAGE_API_KEY', ''))
    base_url: str = 'https://www.alphavantage.co/query'
    output_size: str = 'full'
    request_timeout: int = 20
    request_spacing_sec: float = 0.8
    max_retries: int = 4
    retry_delay: float = 2.0

    cache_dir: str = './av_cache'
    cache_ttl_hours: int = 12
    use_cache: bool = True


class CacheManager:
    """File-based parquet + JSON-metadata cache with TTL."""

    def __init__(self, config: AlphaVantageConfig):
        self.enabled = config.use_cache
        self.cache_dir = Path(config.cache_dir)
        self.ttl = timedelta(hours=config.cache_ttl_hours)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> Tuple[Path, Path]:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f'{h}.parquet', self.cache_dir / f'{h}.json'

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if not self.enabled:
            return None
        data_path, meta_path = self._paths(key)
        if not (data_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            stamped = datetime.fromisoformat(meta['stamped_at'])
            if datetime.now() - stamped > self.ttl:
                return None
            return pd.read_parquet(data_path)
        except Exception:
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        if not self.enabled or df is None or df.empty:
            return
        data_path, meta_path = self._paths(key)
        try:
            df.to_parquet(data_path)
            meta_path.write_text(json.dumps({
                'key': key,
                'stamped_at': datetime.now().isoformat(),
                'rows': len(df),
            }))
        except Exception as e:
            print(f'  [WARN] cache write failed for {key}: {e}')


class AlphaVantageClient:
    """Minimal Alpha Vantage daily OHLCV client."""

    def __init__(self, config: AlphaVantageConfig, cache: Optional[CacheManager] = None):
        if not config.api_key:
            raise ValueError('Alpha Vantage API key is required.')
        self.config = config
        self.cache = cache if cache is not None else CacheManager(config)
        self._last_request_ts: float = 0.0

    def _throttle(self) -> None:
        delta = time.time() - self._last_request_ts
        if delta < self.config.request_spacing_sec:
            time.sleep(self.config.request_spacing_sec - delta)
        self._last_request_ts = time.time()

    def _request(self, params: Dict[str, str]) -> Dict:
        params = dict(params)
        params['apikey'] = self.config.api_key
        last_err: Optional[str] = None
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                resp = requests.get(
                    self.config.base_url,
                    params=params,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                if 'Note' in payload or 'Information' in payload:
                    # Rate-limit / throttling messages
                    last_err = payload.get('Note') or payload.get('Information')
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                if 'Error Message' in payload:
                    raise RuntimeError(payload['Error Message'])
                return payload
            except (requests.RequestException, ValueError) as e:
                last_err = str(e)
                time.sleep(self.config.retry_delay * (2 ** attempt))
        raise RuntimeError(f'Alpha Vantage request failed: {last_err}')

    @staticmethod
    def _parse_time_series(payload: Dict) -> pd.DataFrame:
        series_key = next(
            (k for k in payload.keys() if k.startswith('Time Series')),
            None,
        )
        if series_key is None:
            raise RuntimeError(
                f'No time series in payload (keys={list(payload.keys())})'
            )

        rows = []
        for date_str, bar in payload[series_key].items():
            adj = bar.get('5. adjusted close') or bar.get('4. close')
            rows.append({
                'Date': date_str,
                'Open': float(bar.get('1. open', 'nan')),
                'High': float(bar.get('2. high', 'nan')),
                'Low': float(bar.get('3. low', 'nan')),
                'Close': float(bar.get('4. close', 'nan')),
                'AdjClose': float(adj) if adj is not None else float('nan'),
                'Volume': float(bar.get('6. volume') or bar.get('5. volume') or 0),
            })
        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date').sort_index()

    def get_daily(self, symbol: str) -> pd.DataFrame:
        """Fetch the full daily OHLCV history for a single symbol."""
        cache_key = f'av_daily::{symbol}::{self.config.output_size}'
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        for function in ('TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_DAILY'):
            try:
                payload = self._request({
                    'function': function,
                    'symbol': symbol,
                    'outputsize': self.config.output_size,
                    'datatype': 'json',
                })
                df = self._parse_time_series(payload)
                if df.empty:
                    continue
                self.cache.put(cache_key, df)
                return df
            except RuntimeError as e:
                print(f'  [WARN] {function} failed for {symbol}: {e}')
                continue
        raise RuntimeError(f'No Alpha Vantage data returned for {symbol}')

    def get_many(self, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
        """Fetch many symbols sequentially (respects throttling)."""
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            print(f'  [AV] downloading {sym} …')
            out[sym] = self.get_daily(sym)
        return out

    def download(
        self,
        symbols: Iterable[str],
        field: str = 'AdjClose',
    ) -> pd.DataFrame:
        """yfinance-style wide DataFrame: index=Date, columns=tickers.

        Parameters
        ----------
        symbols : iterable of str
            Tickers to pull.
        field : str
            Column to pivot on ('AdjClose' by default; also 'Close',
            'Open', 'High', 'Low', 'Volume').
        """
        frames: List[pd.Series] = []
        for sym, df in self.get_many(symbols).items():
            if field not in df.columns:
                raise KeyError(f'{sym}: column {field!r} not in {list(df.columns)}')
            frames.append(df[field].rename(sym))
        return pd.concat(frames, axis=1).sort_index()


def download(
    symbols: Iterable[str],
    api_key: str,
    field: str = 'AdjClose',
    **config_overrides,
) -> pd.DataFrame:
    """yfinance-style convenience wrapper.

    >>> from alpha_vantage_live import download
    >>> px = download(['GLDM', 'TLT'], api_key='YOUR_KEY')
    """
    cfg = AlphaVantageConfig(api_key=api_key, **config_overrides)
    client = AlphaVantageClient(cfg)
    return client.download(symbols, field=field)
