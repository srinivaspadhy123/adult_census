import os
import sys
import pandas as pd
import numpy as np
from src.exceptions.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")
            target_path = os.path.join("artifacts","target_preprocessor.pkl")
            preprocessor = load_object(preprocessor_path)
            target_preprocessor = load_object(target_path)
            model = load_object(model_path)
            scaled_feature = preprocessor.transform(features)
            pred = model.predict(scaled_feature)
            actual_pred = target_preprocessor.inverse_transform(pred.reshape(-1,1))

            return actual_pred
        except Exception as exp:
            raise CustomException(exp,sys)
        
class CustomData:
    def __init__(self,age,workclass,fnlwgt,education_num,marital_status,occupation,relationship,
                 race,sex,capital_gain,
                 capital_loss,hours_per_week,country,education):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.country = country
        self.education = education
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'workclass':[self.workclass],
                'fnlwgt':[self.fnlwgt],
                'education-num':[self.education_num],
                'marital-status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'race':[self.race],
                'sex':[self.sex],
                'capital-gain':[self.capital_gain],
                'capital-loss':[self.capital_loss],
                'hours-per-week':[self.hours_per_week],
                'country':[self.country],
                'education':[self.education]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as exp:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(exp,sys)