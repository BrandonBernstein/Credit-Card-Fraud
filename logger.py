import logging
import os
from datetime import datetime

CUSTOM_PATH = "{}.log".format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
LOG_PATH = os.path.join(os.getcwd(),"logs",CUSTOM_PATH)
logger_path = os.path.join(LOG_PATH,CUSTOM_PATH)

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

logging.basicConfig(filename = logger_path,
                    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level = "INFO")

