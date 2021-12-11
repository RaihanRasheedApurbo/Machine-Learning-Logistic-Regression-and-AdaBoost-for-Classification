import pandas as pd
import numpy as np
import logging
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix

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

# global variables
RANDOM_SEED = 7
TEST_DATA_FRACTION = 0.2


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

    np.random.seed(RANDOM_SEED)

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


def performance_measure(y_predicted, y_real):

    accuracy = (y_real == y_predicted).mean()
    logger.warning(f"accuracy: {accuracy}")

    tn, fp, fn, tp = confusion_matrix(y_real, y_predicted).ravel()
    logger.warning(f"tp:{tp},        fn:{fn}")
    logger.warning(f"fp:{fp},        tn:{tn}")

    logger.warning(
        f"True positive rate (sensitivity, recall, hit rate): {(tp/(tp+fn))}"
    )
    logger.warning(f"True negative rate (specificity): {(tn/(tn+fp))}")
    logger.warning(f"Positive predictive value (precision): {(tp/(tp+fp))}")
    logger.warning(f"False discovery rate: {(fp/(tp+fp))}")
    logger.warning(f"F1 score: {(2*tp/(2*tp+fp+fn))}")


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

    # converting  from yes/no to 1/0
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

    # splitting data for test and training
    training_data, testing_data = train_test_split(
        df, test_size=TEST_DATA_FRACTION, random_state=RANDOM_SEED
    )

    output_of_training_data = training_data.Churn.values
    del training_data["Churn"]
    output_of_testing_data = testing_data.Churn.values
    del testing_data["Churn"]

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

    logger.info(type(training_data_feature_matrix_x))
    logger.info(type(output_of_training_data))
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
    performance_measure(churn, y)

    # doing catagorical value encoding using skitlearn's DictVectorizer class
    testing_data_dict = testing_data[categorical_columns + numerical_columns].to_dict(
        orient="records"
    )
    dictionary_vectorizer = DictVectorizer(sparse=False)
    dictionary_vectorizer.fit(testing_data_dict)
    testing_data_feature_matrix_x = dictionary_vectorizer.transform(testing_data_dict)
    logger.info(dictionary_vectorizer.get_feature_names_out())
    logger.info(len(dictionary_vectorizer.get_feature_names_out()))
    logger.info(training_data_feature_matrix_x[0])

    logger.info(type(testing_data_feature_matrix_x))
    logger.info(type(output_of_testing_data))
    x_test = testing_data_feature_matrix_x
    y_test = output_of_testing_data
    logger.info(x_test.shape)
    logger.info(y_test.shape)
    logger.info(np.bincount(y_test))

    churn = model.predict(x_test)
    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y_test))
    performance_measure(churn, y_test)

    majority_model = adaboost(x, y, Logistic_Regression, k=20)
    churn = majority_model.predict(x)

    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y))

    result = (y == churn).mean()
    logger.warning(f"adaboost training data result: {result}")

    churn = majority_model.predict(x_test)

    logger.info(churn)

    logger.info(np.bincount(churn))
    logger.info(np.bincount(y_test))
    result = (y_test == churn).mean()
    logger.warning(f"adaboost test data result: {result}")


def dataset_3():
    df = pd.read_csv("creditcard.csv")
    logger.info(df.head())
    logger.info(df.info())
    logger.info(df["Class"].value_counts())

    # checking and removing duplicates
    logger.info(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    logger.info(df.duplicated().sum())

    # as positive data is scarce hence sampling negetive datas
    # and concatanating two class
    positive_df = df[df.Class == 1]
    negetive_df = df[df.Class == 0]
    sampled_negetive_df = negetive_df.sample(n=10000, random_state=RANDOM_SEED)
    sampled_df = pd.concat([positive_df, sampled_negetive_df])
    logger.info(sampled_df.to_numpy().shape)

    # removing unnecessary columns and forming input x and y
    x = sampled_df.drop(["Class", "Time"], axis=1)
    y = sampled_df["Class"]
    logger.info(x.to_numpy().shape)
    logger.info(y.to_numpy().shape)

    # normalizing numerical attributes
    for attribute in x.columns:
        min_value = x[attribute].min()
        max_value = x[attribute].max()
        x[attribute] = (x[attribute] - min_value) / (max_value - min_value)
    logger.info(x)
    logger.info(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_DATA_FRACTION, random_state=RANDOM_SEED
    )
    logger.info(x_train.shape)
    logger.info(x_test.shape)
    logger.info(y_train.shape)
    logger.info(y_test.shape)

    # converting y values to numpy arr
    # x values gets converted when we create feature matrix
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # doing catagorical value encoding using skitlearn's DictVectorizer class
    training_data_dict = x_train.to_dict(orient="records")
    dictionary_vectorizer = DictVectorizer(sparse=False)
    dictionary_vectorizer.fit(training_data_dict)
    training_data_feature_matrix_x = dictionary_vectorizer.transform(training_data_dict)
    logger.info(dictionary_vectorizer.get_feature_names_out())
    logger.info(len(dictionary_vectorizer.get_feature_names_out()))
    logger.info(training_data_feature_matrix_x[0])

    logger.info(type(training_data_feature_matrix_x))
    logger.info(type(y_train))
    x_train = training_data_feature_matrix_x

    logger.info(x_train.shape)
    logger.info(y_train.shape)
    logger.info(np.bincount(y))

    model = Logistic_Regression(x_train, y_train, alpha=0.0005, max_iterations=1000)
    logger.info(model.co_eff)
    prediction = model.predict(x_train)
    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_train))
    performance_measure(prediction, y_train)

    test_data_dict = x_test.to_dict(orient="records")
    dictionary_vectorizer = DictVectorizer(sparse=False)
    dictionary_vectorizer.fit(test_data_dict)
    testing_data_feature_matrix_x = dictionary_vectorizer.transform(test_data_dict)
    logger.info(dictionary_vectorizer.get_feature_names_out())
    logger.info(len(dictionary_vectorizer.get_feature_names_out()))
    logger.info(testing_data_feature_matrix_x[0])

    logger.info(type(testing_data_feature_matrix_x))
    logger.info(type(y_test))
    x_test = testing_data_feature_matrix_x

    prediction = model.predict(x_test)
    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_test))
    performance_measure(prediction, y_test)

    majority_model = adaboost(x_train, y_train, Logistic_Regression, k=20)
    prediction = majority_model.predict(x_train)

    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_train))

    result = (y_train == prediction).mean()
    logger.warning(f"adaboost training data result: {result}")

    prediction = majority_model.predict(x_test)

    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_test))
    result = (y_test == prediction).mean()
    logger.warning(f"adaboost test data result: {result}")


def dataset_2():
    df1 = pd.read_csv("adult.data", header=None)
    df2 = pd.read_csv("adult.test", header=None)
    # continuous 0,2,4,10,11,12
    # discrete 1,3,5,6,7,8,9,13
    # 0 age: continuous.
    # 1 workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    # 2 fnlwgt: continuous.
    # 3 education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    # 4 education-num: continuous.
    # 5 marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    # 6 occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    # 7 relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    # 8 race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    # 9 sex: Female, Male.
    # 10 capital-gain: continuous.
    # 11 capital-loss: continuous.
    # 12 hours-per-week: continuous.
    # 13 native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    # 14 <=50 or >50

    def clean_dataframe(df):
        logger.info(df.head())
        logger.info(df.info())
        logger.info(df.loc[[0]].to_dict(orient="records"))
        logger.info(df[df.columns].nunique())

        logger.info(df.duplicated().sum())
        df.drop_duplicates(inplace=True)
        logger.info(df.duplicated().sum())

        # we have only one data point for holand-netherlands
        # but no test data for this so
        # so our test feature column will have one less column and therefore
        # won't work properly
        # taking easy way out by romving this data point

        logger.info(df[df[13] == " Holand-Netherlands"])
        df = df[df[13] != " Holand-Netherlands"]
        logger.info(df[df[13] == " Holand-Netherlands"])

        df[14] = df[14].replace(to_replace=" <=50K", value=0)
        df[14] = df[14].replace(to_replace=" <=50K.", value=0)
        df[14] = df[14].replace(to_replace=" >50K", value=1)
        df[14] = df[14].replace(to_replace=" >50K.", value=1)
        logger.info(df[14])
        logger.info(df[14].nunique())

        catagorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
        numerical_columns = [0, 2, 4, 10, 11, 12]

        for col in catagorical_columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].str.lower()

        # replacing question mark with mode of that column
        logger.info(df[df[1] == "?"])
        df_1_mode = df[1].mode()[0]
        df_6_mode = df[6].mode()[0]
        df_13_mode = df[13].mode()[0]
        logger.info(df[df[1] == df_1_mode])
        df[1] = df[1].replace(to_replace="?", value=df_1_mode)
        df[6] = df[6].replace(to_replace="?", value=df_6_mode)
        df[13] = df[13].replace(to_replace="?", value=df_13_mode)
        logger.info(df[df[1] == df_1_mode])

        # normalizing numerical attributes
        for attribute in numerical_columns:
            min_value = df[attribute].min()
            max_value = df[attribute].max()
            df[attribute] = (df[attribute] - min_value) / (max_value - min_value)

        logger.info(df)

        output_of_training_data = df[14].values
        del df[14]

        # doing catagorical value encoding using skitlearn's DictVectorizer class
        training_data_feature_matrix_x = pd.get_dummies(df)
        logger.info(training_data_feature_matrix_x.shape)
        logger.info(training_data_feature_matrix_x[0])

        logger.info(type(training_data_feature_matrix_x))
        logger.info(type(output_of_training_data))

        logger.info(list(training_data_feature_matrix_x.columns))
        # '13_holand-netherlands'
        # logger.info(training_data_feature_matrix_x[training_data_feature_matrix_x['13_holand-netherlands']==1])

        return training_data_feature_matrix_x.to_numpy(), output_of_training_data

    x_train, y_train = clean_dataframe(df1)
    x_test, y_test = clean_dataframe(df2)

    logger.info(x_train.shape)
    logger.info(y_train.shape)
    logger.info(np.bincount(y_train))

    model = Logistic_Regression(x_train, y_train)
    logger.info(model.co_eff)
    prediction = model.predict(x_train)
    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_train))
    performance_measure(prediction, y_train)

    logger.info(x_test.shape)
    logger.info(y_test.shape)
    # logger.info(np.bincount(y_test))

    prediction = model.predict(x_test)
    logger.info(prediction)

    logger.info(np.bincount(prediction))
    # logger.info(np.bincount(y_test))
    performance_measure(prediction, y_test)

    majority_model = adaboost(x_train, y_train, Logistic_Regression, k=20)
    prediction = majority_model.predict(x_train)

    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_train))

    result = (y_train == prediction).mean()
    logger.warning(f"adaboost training data result: {result}")

    prediction = majority_model.predict(x_test)

    logger.info(prediction)

    logger.info(np.bincount(prediction))
    logger.info(np.bincount(y_test))
    result = (y_test == prediction).mean()
    logger.warning(f"adaboost test data result: {result}")


dataset_3()
