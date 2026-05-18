"""
    Handles info and debug logging differently.
    Debug message contains function name and a line
"""
import logging

class CustomFormatter(logging.Formatter):

    info_fmt = "%(asctime)s | %(levelname)s | %(message)s"
    debug_fmt = "%(asctime)s | %(filename)s:%(funcName)s:%(lineno)d | %(levelname)s | %(message)s"
    # debug_fmt = "%(asctime)s | %(module)s | %(filename)s:%(funcName)s:%(lineno)d | %(levelname)s | %(message)s"

    def format(self, record):
        if record.levelno == logging.DEBUG:
            formatter = logging.Formatter(self.debug_fmt)
        else:
            formatter = logging.Formatter(self.info_fmt)

        return formatter.format(record)


# === Usage ===
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler("app.log")
# file_handler.setFormatter(CustomFormatter())

# logger.addHandler(file_handler)
# logger.info("Training started")
# logger.debug("Batch loss = %f", 0.23)

# === Output ===
# 2026-03-06 19:05:01 | INFO | Training started
# 2026-03-06 19:05:02 | train.py:train_step:42 | DEBUG | Batch loss = 0.23