import logging
import sys

#Logging configuration
def configure_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout