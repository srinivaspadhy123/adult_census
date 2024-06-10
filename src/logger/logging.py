import logging
from datetime import datetime
import os

file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
file_path = os.path.join(os.getcwd(),"logs")
os.makedirs(file_path)
FILEPATH = os.path.join(file_path,file_name)

logging.basicConfig(
    level=logging.INFO,
    filename=FILEPATH,
    format = '[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    logging.info("log testing")