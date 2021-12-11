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

regression_logger = logging.getLogger("Linear_Regression")
regression_logger.setLevel(logging.ERROR)
regression_logger.addHandler(stream_handler)


class Logistic_Regression:
    def __init__(self, x_values, y_values, alpha=0.00005, max_iterations=200) -> None:
        if len(x_values) != len(y_values):
            raise Exception

        # y has value 0 for no and 1 for yes
        # for tanh to work we have to decode no as -1
        # that's why replacing value 0 with -1

        self.x = x_values.copy()
        self.y = y_values.copy()
        regression_logger.info(self.y)
        self.y[self.y == 0] = -1
        regression_logger.info(self.y)
        self.alpha = alpha
        self.max_iterations = max_iterations

        number_of_samples, number_of_features = self.x.shape
        number_of_co_efficients = number_of_features + 1  # 1 extra for Wo
        one_column = np.ones((number_of_samples, 1))
        regression_logger.info(self.x.shape)
        self.x = np.append(self.x, one_column, axis=1)
        regression_logger.info(self.x.shape)
        self.co_eff = [0] * number_of_co_efficients
        self.begin_training()

    def begin_training(self):

        for cycle in range(self.max_iterations):
            h_x = self.hypothesis_of_x(self.x)  # finidng hw(x)
            # regression_logger.info(np.bincount(h_x))
            # alpha * (y-h(x)) * (1-h(x)^2) * xi + w = w
            # |       multiplier          |
            # |       update                    |
            multiplier = self.alpha * (self.y - h_x) * (1 - h_x * h_x)
            regression_logger.info(len((1 - h_x * h_x)))
            regression_logger.info((1 - h_x * h_x))
            regression_logger.info(len(self.y))
            regression_logger.info(len(h_x))
            regression_logger.info((self.y - h_x))
            regression_logger.info(((self.y - h_x) * (1 - h_x * h_x)))

            regression_logger.info(multiplier)
            regression_logger.info(multiplier)
            matrix_multiplier = multiplier.transpose()
            regression_logger.info(matrix_multiplier)
            matrix_x = self.x
            regression_logger.info(matrix_x.shape)
            regression_logger.info(matrix_multiplier.shape)
            update = np.matmul(matrix_multiplier, matrix_x)
            regression_logger.info(update)
            regression_logger.info(update.shape)
            new_co_eff = self.co_eff + update
            regression_logger.info(new_co_eff)

            # finding average difference for co_effs
            total_difference = np.sum(abs(np.array(new_co_eff) - np.array(self.co_eff)))
            regression_logger.info(total_difference)
            # difference = 0
            # for i in range(len(new_co_eff)):
            #     difference += abs(new_co_eff[i]-self.co_eff[i])
            average_difference = total_difference / len(new_co_eff)
            # if average difference is small then break
            # if(average_difference<0.5):
            #     regression_logger.info(f"breaking in cycle no {cycle}")
            #     break
            # updating co_effs
            self.co_eff = new_co_eff
            regression_logger.warning(
                f"after cycle no {cycle} co_effs are {self.co_eff}"
            )

    @staticmethod
    def apply_tanh(z):
        e = 0.05
        tanh_value = math.tanh(z)
        if tanh_value < -0.95:
            tanh_value = tanh_value + e
        elif tanh_value > 0.95:
            tanh_value = tanh_value - e
        return tanh_value

    def hypothesis_of_x(self, x):
        matrix_x = x
        matrix_co_eff = np.array(self.co_eff).transpose()
        regression_logger.info(matrix_x)
        regression_logger.info(matrix_co_eff)
        h_x = np.matmul(matrix_x, matrix_co_eff)
        regression_logger.info(h_x)
        regression_logger.info(h_x.shape)
        vfunc = np.vectorize(Logistic_Regression.apply_tanh)
        h_x = vfunc(h_x)
        # h_x = np.apply_along_axis(Logistic_Regression.apply_tanh,0,h_x.transpose())
        regression_logger.info(h_x)
        # df = pd.DataFrame(h_x)
        # df[0] = df[0].apply(Logistic_Regression.apply_tanh, 0)
        # regression_logger.info(df)
        return h_x

    def predict(self, x):
        x_with_xo = x.copy()
        regression_logger.info(x_with_xo)
        number_of_samples, number_of_features = x_with_xo.shape
        one_column = np.ones((number_of_samples, 1))
        regression_logger.info(x_with_xo)
        x_with_xo = np.append(x_with_xo, one_column, axis=1)
        regression_logger.info(x_with_xo)
        tanh_hypothesis_value = self.hypothesis_of_x(x_with_xo)
        regression_logger.info(tanh_hypothesis_value)
        prediction = tanh_hypothesis_value >= 0
        regression_logger.info(prediction)
        prediction = prediction.astype(int)
        regression_logger.info(prediction)
        return prediction


class Weighted_Majority:
    weighted_majority_logger = logging.getLogger("Weighted_Majority")
    weighted_majority_logger.setLevel(logging.WARNING)
    weighted_majority_logger.addHandler(stream_handler)

    def __init__(self, hypothesis_group, hypothesis_weights) -> None:
        logger = Weighted_Majority.weighted_majority_logger
        if len(hypothesis_group) != len(hypothesis_weights):
            logger.error(f"{len(hypothesis_group)} {len(hypothesis_weights)}")
            raise Exception
        self.hypothesis_group = hypothesis_group
        self.hypothesis_weights = hypothesis_weights

    def predict(self, x):
        logger = Weighted_Majority.weighted_majority_logger
        result = np.zeros(len(x))
        for i in range(len(self.hypothesis_group)):
            prediction = self.hypothesis_group[i].predict(x)
            # df has 0/1 as output we are converting it to -1/1
            prediction[prediction == 0] = -1
            logger.info(self.hypothesis_group[i].co_eff)
            logger.info((prediction))
            logger.info(self.hypothesis_weights[i])
            result = result + prediction * self.hypothesis_weights[i]
            logger.info(result)
        prediction = result >= 0
        logger.info(prediction)
        prediction = prediction.astype(int)
        logger.info(prediction)
        return prediction


def adaboost(x, y, L, k=5):
    adaboost_logger = logging.getLogger("AdaBoost")
    adaboost_logger.setLevel(logging.WARNING)
    adaboost_logger.addHandler(stream_handler)

    example_count = len(x)
    hypothesis_weights = [0] * k
    hypothesis_group = []
    index_array = np.arange(example_count)
    prob_array = np.full(example_count, 1 / example_count)
    adaboost_logger.info(index_array)
    adaboost_logger.info(prob_array)

    seed_value = 5
    np.random.seed(seed_value)

    for i in range(k):
        sampled_index = np.random.choice(index_array, size=example_count, p=prob_array)
        adaboost_logger.info(sampled_index)
        adaboost_logger.info(np.bincount(sampled_index))
        sampled_x = x[sampled_index, :]
        sampled_y = y[sampled_index]
        adaboost_logger.info(sampled_x.shape)
        adaboost_logger.info(sampled_y.shape)

        model = L(sampled_x, sampled_y)
        hypothesis_group.append(model)
        prediction = model.predict(x)
        logger.info(np.bincount(prediction))
        adaboost_logger.info(x.shape)
        adaboost_logger.info(len(model.co_eff))
        adaboost_logger.info(model.co_eff)

        adaboost_logger.info(np.bincount(prediction))
        result = (y == prediction).astype(int)
        adaboost_logger.info(result)
        # we want to figure out which predictions are wrong so we are going to invert result
        # (1 - result) flips the bit so 1 present we had our prediction wrong here
        result_bar = 1 - result
        adaboost_logger.info(result_bar)
        error = result_bar * prob_array
        adaboost_logger.info(error)
        error_sum = error.sum()
        adaboost_logger.info(error_sum)
        if error_sum > 0.5:
            continue
        # reducing weight from 1 to below 1 for right predicted data points
        probability_update_multiplier = result * (error_sum / (1 - error_sum))
        # wrong predicted rows have 0 in them giving them weight of 1
        probability_update_multiplier[probability_update_multiplier == 0] = 1
        adaboost_logger.info(probability_update_multiplier)

        # updating probability
        prob_array = prob_array * probability_update_multiplier
        adaboost_logger.info(prob_array)

        # normalizing
        prob_sum = prob_array.sum()
        prob_array = prob_array / prob_sum
        adaboost_logger.info(prob_array)

        hypothesis_weights[i] = math.log((1 - error_sum) / error_sum)

    adaboost_logger.info(hypothesis_weights)
    return Weighted_Majority(hypothesis_group, hypothesis_weights)


def dataset_1():

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    logger.info(len(df))
    logger.info(df.head())
    logger.info(df.dtypes)
    logger.info(df[df.columns].isnull().sum())
    logger.info(df[df.columns].nunique())

    categorical_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    # TotalCharges column has float value but it is parsed as an object
    # that's is why we are converting it to numeric float64
    total_charges = pd.to_numeric(
        df.TotalCharges, errors="coerce"
    )  # converting to float64
    logger.info(total_charges)
    logger.info(type(total_charges))
    # logging blank valued rows after converting to numeric
    logger.info(
        df[total_charges.isnull()][["customerID", "TotalCharges", "MonthlyCharges"]]
    )
    df["TotalCharges"] = total_charges

    # replacing blanks with the MonthlyCharges attribute value of same row
    values = df[total_charges.isnull()]["MonthlyCharges"].values
    df.loc[total_charges.isnull(), "TotalCharges"] = values
    logger.info(
        df[total_charges.isnull()][["customerID", "TotalCharges", "MonthlyCharges"]]
    )

    # converting Y/Output value from yes/no to 1/-1
    logger.info(df.Churn.head())
    logger.info(df.Churn.value_counts())
    df.Churn = df["Churn"].replace(to_replace="Yes", value=1)
    df.Churn = df["Churn"].replace(to_replace="No", value=0)
    logger.info(df.Churn.head())
    logger.info(df.Churn.value_counts())

    # normalizing numerical attributes
    for attribute in numerical_columns:
        min_value = df[attribute].min()
        max_value = df[attribute].max()
        df[attribute] = (df[attribute] - min_value) / (max_value - min_value)
    # logging after normalization
    logger.info(df[["customerID"] + numerical_columns])

    # using same random_seed variable for reproducible result
    random_seed = 7
    test_data_fraction = 0.2
    validation_data_fraction = 0.33

    # splitting data for test and training
    full_training_data, testing_data = train_test_split(
        df, test_size=test_data_fraction, random_state=random_seed
    )
    # splitting full training data into training and validation data
    training_data, validation_data = train_test_split(
        full_training_data, test_size=validation_data_fraction, random_state=random_seed
    )

    # storing output churn column in sperate place and deleting this column from the dataframe
    output_of_training_data = training_data.Churn.values
    output_of_validation_data = validation_data.Churn.values
    del training_data["Churn"]
    del validation_data["Churn"]

    logger.info(full_training_data.Churn.value_counts())

    # doing catagorical value encoding using skitlearn's DictVectorizer class
    training_data_dict = training_data[categorical_columns + numerical_columns].to_dict(
        orient="records"
    )
    dictionary_vectorizer = DictVectorizer(sparse=False)
    dictionary_vectorizer.fit(training_data_dict)
    training_data_feature_matrix_x = dictionary_vectorizer.transform(training_data_dict)
    logger.info(dictionary_vectorizer.get_feature_names_out())
    logger.info(len(dictionary_vectorizer.get_feature_names_out()))
    logger.info(training_data_feature_matrix_x[0])

    # test dummy data
    # 1231  147 w1
    # 4561  258 w2
    # 7891  369 w3
    #       111
    # data_x = [[1,2,3],[4,5,6],[7,8,9]]
    # data_y = [1,1,0]
    # x = pd.DataFrame(data_x)
    # y = pd.DataFrame(data_y)
    logger.info(type(training_data_feature_matrix_x))
    x = training_data_feature_matrix_x
    y = output_of_training_data
    logger.info(x.shape)
    logger.info(y.shape)
    logger.info(np.bincount(y))

    model = Logistic_Regression(x, y)
    logger.info(model.co_eff)
    churn = model.predict(x)
    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y))
    result = (y == churn).mean()
    logger.info(result)

    # doing catagorical value encoding using skitlearn's DictVectorizer class
    validation_data_dict = validation_data[
        categorical_columns + numerical_columns
    ].to_dict(orient="records")
    dictionary_vectorizer = DictVectorizer(sparse=False)
    dictionary_vectorizer.fit(validation_data_dict)
    validation_data_feature_matrix_x = dictionary_vectorizer.transform(
        validation_data_dict
    )
    logger.info(dictionary_vectorizer.get_feature_names_out())
    logger.info(len(dictionary_vectorizer.get_feature_names_out()))
    logger.info(training_data_feature_matrix_x[0])

    logger.info(type(validation_data_feature_matrix_x))
    x_validation = validation_data_feature_matrix_x
    y_validation = output_of_validation_data
    logger.info(x_validation.shape)
    logger.info(y_validation.shape)
    logger.info(np.bincount(y_validation))

    churn = model.predict(x_validation)
    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y_validation))
    result = (y_validation == churn).mean()
    logger.info(result)

    majority_model = adaboost(x, y, Logistic_Regression, 15)
    churn = majority_model.predict(x)

    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y))
    result = (y == churn).mean()
    logger.info(result)

    churn = majority_model.predict(x_validation)

    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y_validation))
    result = (y_validation == churn).mean()
    logger.info(result)


dataset_1()
