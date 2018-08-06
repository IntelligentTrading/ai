import logging

log_level = logging.DEBUG
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

logger.info("ittai LOG Level: {}".format(log_level))