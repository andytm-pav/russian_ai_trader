"""
Пакет для получения данных с различных источников
"""

from .moex_fetcher import MoexFetcher
from .news_fetcher import NewsFetcher
from .rss_fetcher import RSSFetcher

__all__ = ['MoexFetcher', 'NewsFetcher', 'RSSFetcher']