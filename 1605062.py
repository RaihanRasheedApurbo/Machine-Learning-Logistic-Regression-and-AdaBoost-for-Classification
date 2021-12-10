from os import XATTR_SIZE_MAX
import pandas as pd
import numpy as np
import logging
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# logging initialization

formatter = logging.Formatter(
    "\n*********Line no:%(lineno)d*********\n%(message)s\n***************************"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

regression_logger = logging.getLogger('Linear_Regression')
regression_logger.setLevel(logging.WARNING)
regression_logger.addHandler(stream_handler)



df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
logger.info(len(df))
logger.info(df.head())
logger.info(df.dtypes)
logger.info(df[df.columns].isnull().sum())
logger.info(df[df.columns].nunique())



categorical_columns = ['gender','SeniorCitizen','Partner','Dependents','PhoneService',
              'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
              'PaperlessBilling','PaymentMethod']
numerical_columns = [ 'tenure','MonthlyCharges','TotalCharges']




# TotalCharges column has float value but it is parsed as an object
# that's is why we are converting it to numeric float64
total_charges = pd.to_numeric(df.TotalCharges, errors="coerce")  # converting to float64
logger.info(total_charges)
logger.info(type(total_charges))
# logging blank valued rows after converting to numeric
logger.info(
    df[total_charges.isnull()][["customerID", "TotalCharges", "MonthlyCharges"]]
)  
df["TotalCharges"] = total_charges

# replacing blanks with the MonthlyCharges attribute value of same row
values = df[total_charges.isnull()]["MonthlyCharges"].values
df.loc[total_charges.isnull(),"TotalCharges"] = values 
logger.info(
    df[total_charges.isnull()][["customerID", "TotalCharges", "MonthlyCharges"]]
)

# converting Y/Output value from yes/no to 1/-1
logger.info(df.Churn.head())
logger.info(df.Churn.value_counts())
df.Churn = df["Churn"].replace(to_replace="Yes", value = 1)
df.Churn = df["Churn"].replace(to_replace="No", value = -1)
logger.info(df.Churn.head())
logger.info(df.Churn.value_counts())

# normalizing numerical attributes
for attribute in numerical_columns:
    min_value = df[attribute].min()
    max_value = df[attribute].max()
    df[attribute] = (df[attribute]-min_value)/(max_value-min_value)
# logging after normalization
logger.info(
    df[["customerID"]+numerical_columns]
)

# using same random_seed variable for reproducible result
random_seed = 7
test_data_fraction = 0.2
validation_data_fraction = 0.33

# splitting data for test and training
full_training_data, testing_data = train_test_split(df, test_size=test_data_fraction, random_state=random_seed)
# splitting full training data into training and validation data
training_data, validation_data =  train_test_split(full_training_data, test_size=validation_data_fraction, random_state=random_seed)

# storing output churn column in sperate place and deleting this column from the dataframe
output_of_training_data = training_data.Churn.values
output_of_validation_data = validation_data.Churn.values
del training_data['Churn']
del validation_data['Churn']

logger.info(full_training_data.Churn.value_counts())


# doing catagorical value encoding using skitlearn's DictVectorizer class
training_data_dict = training_data[categorical_columns+numerical_columns].to_dict(orient='records')
dictionary_vectorizer = DictVectorizer(sparse=False)
dictionary_vectorizer.fit(training_data_dict)
training_data_feature_matrix_x = dictionary_vectorizer.transform(training_data_dict)
logger.info(dictionary_vectorizer.get_feature_names_out())
logger.info(len(dictionary_vectorizer.get_feature_names_out()))
logger.info(training_data_feature_matrix_x[0])
# 1231  147 w1     
# 4561  258 w2
# 7891  369 w3
#       111 
# data_x = [[1,2,3],[4,5,6],[7,8,9]]
# data_y = [1,1,-1]
# x = pd.DataFrame(data_x)
# y = pd.DataFrame(data_y)

x = pd.DataFrame(training_data_feature_matrix_x)
y = pd.DataFrame(output_of_training_data)
logger.info(y[0].value_counts())



class Logistic_Regression:

    @staticmethod
    def apply_tanh(z):  
        e = 0.05
        tanh_value = math.tanh(z)
        if tanh_value < -.95:
            tanh_value = tanh_value + e  
        elif tanh_value > .95:   
            tanh_value = tanh_value - e      
        return tanh_value

    def hypothesis_of_x(self,x):
        matrix_x = x.to_numpy()
        matrix_co_eff = np.array(self.co_eff).transpose()
        regression_logger.info(matrix_x)
        regression_logger.info(matrix_co_eff)
        h_x = np.matmul(matrix_x,matrix_co_eff)
        regression_logger.info(h_x)
        df = pd.DataFrame(h_x)
        df[0] = df[0].apply(Logistic_Regression.apply_tanh,0)
        regression_logger.info(df)
        return df

    def begin_training(self):
        alpha = 0.00005 
        for cycle in range(100):
            h_x = self.hypothesis_of_x(self.x) # finidng hw(x)
            regression_logger.info(h_x[0].value_counts())
            # alpha * (y-h(x)) * (1-h(x)^2) * xi + w = w
            # |       multiplier          |
            # |       update                    |
            multiplier = alpha * (self.y-h_x) * (1-h_x*h_x)
            regression_logger.info((self.y-h_x)[0].value_counts())
            regression_logger.info(multiplier)
            regression_logger.info(multiplier[0].value_counts())
            regression_logger.info(multiplier)
            matrix_multiplier = multiplier.to_numpy().transpose()
            regression_logger.info(matrix_multiplier)
            matrix_x = self.x.to_numpy()
            regression_logger.info(matrix_x)
            update = np.matmul(matrix_multiplier,matrix_x)
            regression_logger.info(update)
            new_co_eff = (self.co_eff + update)
            regression_logger.info(new_co_eff)
            # new_co_eff is a 2D matrix with only one row.
            # so we are converting it to a python list
            new_co_eff = new_co_eff[0].tolist() 
            regression_logger.info(new_co_eff)

            #finding average difference for co_effs
            total_difference = np.sum(abs(np.array(new_co_eff)-np.array(self.co_eff)))
            regression_logger.info(total_difference)
            # difference = 0
            # for i in range(len(new_co_eff)):
            #     difference += abs(new_co_eff[i]-self.co_eff[i])
            average_difference = total_difference/len(new_co_eff)
            # if average difference is small then break
            # if(average_difference<0.5):
            #     regression_logger.info(f"breaking in cycle no {cycle}")
            #     break
            # updating co_effs
            self.co_eff = new_co_eff
            regression_logger.warning(f"after cycle no {cycle} co_effs are {self.co_eff}")

    def __init__(self,x_values,y_values) -> None:
        if len(x_values) != len(y_values):
            raise Exception
        self.x = x_values
        self.y = y_values
        number_of_features = len(self.x.columns)
        number_of_co_efficients = number_of_features + 1 # 1 extra for Wo
        self.x[len(self.x.columns)] = 1 # adding Xo = 1 as the dummy variable 
        regression_logger.info(self.x)
        self.co_eff = [0] * number_of_co_efficients
        self.begin_training()
    
    def predict(self,x,y):
        df = self.hypothesis_of_x(x)
        regression_logger.warning(df.head())
        regression_logger.warning(y.head())
        churn = df >= 0
        regression_logger.warning(churn)
        regression_logger.warning(churn.value_counts())

        churn = churn.replace(to_replace=True, value = 1)
        churn = churn.replace(to_replace=False, value = -1)
        regression_logger.warning(churn)
        regression_logger.warning(churn.value_counts())
        result = (y == churn).mean()
        regression_logger.warning(result)

model = Logistic_Regression(x,y)    
logger.info(model.co_eff)
model.predict(x,y)



