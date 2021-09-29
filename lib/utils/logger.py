import logging
import os
import sys

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    # don't log results for the non-master process
    if distributed_rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    logger.setLevel(logging.DEBUG)
    #ch = logging.StreamHandler(stream=sys.stdout)
    #ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    #logger.setFormatter(formatter)
    # logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
