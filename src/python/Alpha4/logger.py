"""
Helper functions to set up formatted logging to console with level info.
File logger can be added with level debug.

Author: AdamP 2020-2020
"""

import logging
from logging.handlers import RotatingFileHandler


def _get_formatter():
    return logging.Formatter('%(asctime)s --%(levelname)s-- %(message)s')


def get_logger():
    """Returns formatted info logger to console"""
    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    log_formatter = _get_formatter()
    # define a Handler which writes INFO messages or higher to console
    log_console = logging.StreamHandler()
    log_console.setFormatter(log_formatter)
    log_console.setLevel(logging.INFO)
    log.addHandler(log_console)
    return log


def add_file_logger(log, log_path):
    """Add logging to file with debug level to existing log"""
    # set up logging to file
    log_file = logging.handlers.RotatingFileHandler(log_path, 'a', maxBytes=1024*1024, backupCount=1024*1024)
    log_file.setFormatter(_get_formatter())
    log_file.setLevel(logging.DEBUG)
    log.addHandler(log_file)
