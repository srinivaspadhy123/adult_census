import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from dataclasses import dataclass
from src.utils.utils import save_object,load_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
    target_preprocessor_path = os.path.join("artifacts","target_preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            cat_cols = ['workclass',
                'education',
                'marital-status',
                'relationship',
                'race',
                'sex',
                'country',
                'occupation']
            
            num_cols = ['age',
                'fnlwgt',
                'education-num',
                'capital-gain',
                'capital-loss',
                'hours-per-week']
            
            
            
            # Define the custom ranking for each ordinal variable           
            occupation_categories = ['Exec-managerial','Prof-specialty','Sales','Craft-repair','Adm-clerical','Transport-moving','Tech-support',
                        'Machine-op-inspct','Protective-serv','Farming-fishing','Handlers-cleaners','Other-service','Priv-house-serv','Armed-Forces']
            
            logging.info('Pipeline Initiated')
            
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            # Categorical Pipeline for Ordinal Encoding
            # cat_pipeline_ordinal = Pipeline(
            #     steps=[
            #         ('imputer',SimpleImputer(strategy='most_frequent')),
            #         ('ordinal_encoder',OrdinalEncoder(categories=[occupation_categories]))
            #     ]
            # )

            # Categorigal Pipeline
            cat_pipeline_oh = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('oh_encoder',OneHotEncoder(sparse_output=False))
                ]
            )
            salary_categories = ['<=50K','>50K']
            # target_cat_pipeline = Pipeline(
            #     steps=[
            #         ('imputer',SimpleImputer(strategy='most_frequent')),
            #         ('oh_encoder',OrdinalEncoder(categories=[salary_categories]))
            #     ]
            # )
            
            # Column Transformer for the pipeline transformation
            preprocessor = ColumnTransformer([
                    ('num_pipeline',num_pipeline,num_cols),
                    # ('cat_pipeline_ordinal',cat_pipeline_ordinal,['occupation']),
                    ('cat_pipeline_oh',cat_pipeline_oh,cat_cols)
                    
            ])
            
            target_preprocessor = OrdinalEncoder(categories=[salary_categories])
            return preprocessor,target_preprocessor
        except Exception as exp:
            logging.info("Error occured while applying transformation")
            raise CustomException(exp,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data Complete")
            logging.info(f"Train DataFrame Head : \n{train_df.head(5)}")
            logging.info(f"Test DataFrame Head : \n{test_df.head(5)}")

            preprocessing_obj,target_preprocessing_obj = self.get_data_transformation()
            target_column_name = 'salary'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            target_feature_train_df = target_preprocessing_obj.fit_transform(train_df[[target_column_name]])
            target_feature_test_df = target_preprocessing_obj.fit_transform(test_df[[target_column_name]])
            logging.info("Applying Preprocessing Object on Training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]

            save_object(
                file_path=self.preprocessor_obj.preprocessor_path,
                obj=preprocessing_obj
            )
            save_object(
                file_path=self.preprocessor_obj.target_preprocessor_path,
                obj=target_preprocessing_obj
            )
            logging.info("Preprocessing Pkl file saved")
            
            return (
                train_arr,
                test_arr
            )
        except Exception as exp:
            logging.info("Exception Ocurred during data transformation phase")
            raise CustomException(exp,sys)