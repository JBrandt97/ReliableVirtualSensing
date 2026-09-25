import logging
import os

def setup_logging(log_dir, log_level='INFO', filename='training.log'):
    os.makedirs(log_dir, exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    file_path = os.path.join(log_dir, filename)
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return file_handler

def cleanup_logging(file_handler):
    root_logger = logging.getLogger()
    root_logger.removeHandler(file_handler)
    file_handler.close()
