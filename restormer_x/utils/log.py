import logging
import os


def set_logger(log_dir, file_name):
    loglevel = logging.INFO

    log_path = os.path.join(log_dir, file_name)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    logging.info('writting logs to file {}'.format(log_path))
