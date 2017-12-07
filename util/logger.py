# coding='utf-8'

import logging
import numpy as np

from configure import properties


# create a log recorder
def log(file_path, mode='a'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=file_path,
                        filemode=mode)  # a: append, w: re-write
    logger_object = logging.getLogger()
    return logger_object


if __name__ == '__main__':
    logger = log(properties.base_path()+ 'mylog.log')

    logger.debug('This is debug message.')
    logger.info('This is info message')
    logger.warning('This is warning message')
    logger.info({'a': 1, 'b': 2})
