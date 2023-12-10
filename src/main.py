import argparse
import configparser
import logging
import os

import ingest_data
import score
import train

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


def main():
    """Function to add argparse arguments to accept user inputs, configure the logger
    calls the functions to train, score the data

    Parameters
    ----------
        None

    Returns
    -------
        None

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="path to access the hpython maiousing data")
    parser.add_argument(
        "--config_file", help="Specify the path to the configuration file"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )
    parser.add_argument("--log_path", help="Path to the log file")
    parser.add_argument(
        "--no_console_log", action="store_true", help="Disable logging to the console"
    )
    parser.add_argument("output_folder", help="Output folder for datasets and model")

    args = parser.parse_args()
    config = configparser.ConfigParser()

    if args.config_file:
        config.read(args.config_file)
    else:
        config.read("setup.cfg")

    log_level = config.get("Logging", "log_level", fallback="INFO")
    log_path = config.get("Logging", "log_path", fallback=None)
    console_log_not_enabled = config.getboolean(
        "Logging", "console_log_not_enabled", fallback=False
    )

    if args.log_level:
        log_level = args.log_level
    if args.log_path:
        log_path = args.log_path
    if args.no_console_log:
        console_log_not_enabled = True

    logger = configure_logger(
        log_file=log_path, console=console_log_not_enabled, log_level_var=log_level
    )
    logger.warning(log_level)

    housing_data = ingest_data.get_data(args.output_folder)
    (
        housing_prepared,
        housing_labels,
        strat_test_set,
        imputer,
    ) = train.training_data(housing_data, args.output_folder)
    score.score(housing_prepared, housing_labels, strat_test_set, imputer)


if __name__ == "__main__":
    main()
