import sys
import contextlib
import os
import tqdm
import logging

__all__ = [
    "write_log_to_file",
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

def ERROR(string):
    _global_logger.error(string)

def INFO(string):
    _global_logger.info(string)

def WARN(string):
    _global_logger.warning(string)

def PRINT(*string):
    ss = [s if isinstance(s, str) else '{0}'.format(s) for s in string]
    sys.stderr.write('{0}\n'.format(' '.join(ss)))



