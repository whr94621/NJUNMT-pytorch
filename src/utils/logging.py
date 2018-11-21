import contextlib
import logging
import os
import sys

import tqdm

__all__ = [
    "write_log_to_file",
    'set_logging_level',
    'close_logging',
    "INFO",
    "WARN",
    "ERROR",
    "PRINT"
]


class TqdmLoggingHandler(logging.Handler):

    def __init__(self, level=logging.NOTSET):

        super(self.__class__, self).__init__(level)

    def emit(self, record):

        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except(KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def _init_global_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(tqdm_handler)

    return logger


class GlobalLogger(object):
    _GLOBAL_LOGGER = _init_global_logger()

    @staticmethod
    def write_log_to_file(log_file):
        """
        Redirect log information to file as well
        """
        log_dir = os.path.dirname(log_file)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        filer_handler = logging.FileHandler(log_file)
        filer_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        GlobalLogger._GLOBAL_LOGGER.addHandler(filer_handler)

    @staticmethod
    @contextlib.contextmanager
    def global_logging():

        if GlobalLogger._GLOBAL_LOGGER is None:
            print("Global logger is not initialized!")
            raise ValueError

        yield

        pass


_global_logger = GlobalLogger._GLOBAL_LOGGER

write_log_to_file = GlobalLogger.write_log_to_file

LOGGING_LEVEL = 0


def set_logging_level(level):
    global LOGGING_LEVEL
    LOGGING_LEVEL = level


def close_logging():
    global LOGGING_LEVEL
    LOGGING_LEVEL = 60


def ERROR(string):
    if LOGGING_LEVEL <= 40:
        _global_logger.error(string)


def INFO(string):
    if LOGGING_LEVEL <= 20:
        _global_logger.info(string)


def WARN(string):
    if LOGGING_LEVEL <= 30:
        _global_logger.warning(string)


def PRINT(*string):
    if LOGGING_LEVEL <= 20:
        ss = [s if isinstance(s, str) else '{0}'.format(s) for s in string]
        sys.stderr.write('{0}\n'.format(' '.join(ss)))
