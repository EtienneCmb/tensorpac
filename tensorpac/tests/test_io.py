"""Test I/O."""
import logging

from tensorpac.io import (set_log_level, is_pandas_installed,
                          is_statsmodels_installed, progress_bar)

logger = logging.getLogger('tensorpac')


levels = ['profiler', 'debug', 'info', 'warning', 'error', 'critical']


class TestIO(object):
    """Test statistical functions."""

    def test_log_level(self):
        """Test setting the log level."""
        for l in levels:
            set_log_level(l)
        set_log_level(False)
        set_log_level(True)
        set_log_level(match="ok")
        logger.info("show me ok")
        logger.info("show me")

    def test_logger(self):
        """Test logger levels."""
        set_log_level("profiler")
        logger.profiler("profiler")
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.critical("critical")

    def test_progress_bar(self):
        """Test progress bar."""
        progress_bar(5, 10)

    def test_dependance(self):
        """Test dependancies."""
        is_statsmodels_installed()
        is_pandas_installed()
