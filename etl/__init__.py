"""
ETL package for downloading and preprocessing AG News dataset.
"""

from .download_data import download_ag_news
from .preprocess import process_ag_news, clean_text, AG_NEWS_CLASSES
