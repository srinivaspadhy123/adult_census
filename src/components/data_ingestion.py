
import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from sklearn.model_selection import train_test_split
from src.exceptions.exception import CustomException
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_path = os.path.join("artifacts","train.csv")
    test_path = os.path.join("artifacts","test.csv")
    # raw_path = os.path.join("artifacts","raw.csv")


class DataIngestion:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading the Data from the artifacts folder")
            census_data = pd.read_csv("artifacts/adult.csv")
            logging.info("Initiating Train Test Split")
            train_data,test_data = train_test_split(census_data,test_size=0.25)
            logging.info("Train Test Split completed.")
            logging.info("Proceeding for saving the train and test data")
            train_data.to_csv(self.data_ingestion_config.train_path)
            test_data.to_csv(self.data_ingestion_config.test_path)

            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )
        except Exception as exp:
            logging.info("Error Occured while ingesting data")
            raise CustomException(exp,sys)
