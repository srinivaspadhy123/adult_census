import os
import sys
import mlflow
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
from src.utils.utils import evaluate_model,load_object
from src.logger.logging import logging
from src.exceptions.exception import CustomException
from dataclasses import dataclass



class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self,actual,pred):
        accuracy_score = accuracy_score(actual,pred)
        roc_auc_score = roc_auc_score(actual,pred)
        f1_score = f1_score(actual,pred)
        logging.info("Evaluation Metrics Captured")
        return accuracy_score,roc_auc_score,f1_score
    
    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test = (test_array[:,:-1],test_array[:,-1])
            model_path=os.path.join("artifacts","model.pkl")
            model = load_object(model_path)
            # mlflow.set_registry_uri("")
            logging.info("Model has registered")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(X_test)
                (accuracy_score,roc_auc_score,f1_score) = self.eval_metrics(y_test,prediction)
                mlflow.log_metric("accuracy",accuracy_score)
                mlflow.log_metric("roc_auc",roc_auc_score)
                mlflow.log_metric("f1",f1_score)
                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(model,"model",registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model,"model")

        except Exception as exp:
            raise CustomException(exp,sys)

