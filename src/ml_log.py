import logging
import os

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                """%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:
                                    %(lineno)d - %(message)s"""
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "INFO"},
}


def configure_logger(
    cfg=None,
    log_file=None,
    console=False,
    log_level_var="INFO",
):
    """Function to configure the logger

    Parameters
    ----------
    cfg
    log_file
    console
    log_level_var

    Returns
    -------
    logger

    """

    if not logging.getLogger().hasHandlers():
        if not cfg:
            logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
        else:
            logging.config.dictConfig(cfg)

        logger = logging.getLogger()
        logger.setLevel(log_level_var)

        if log_file or console:
            if log_file:
                file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), log_file
                )
                fh = logging.FileHandler(file)
                logger.addHandler(fh)
            if not console:
                sh = logging.StreamHandler()
                logger.addHandler(sh)

        return logger
    else:
        return logging.getLogger()