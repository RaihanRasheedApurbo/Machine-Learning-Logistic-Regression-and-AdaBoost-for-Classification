import pandas as pd
import logging

# logging initialization

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "\n*********Line no:%(lineno)d*********\n%(message)s\n***************************"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
logger.info(len(df))
logger.info(df.head())
logger.info(df.dtypes)


# TotalCharges column has float value but it is parsed as an object
# that's is why we are converting it to numeric float64
total_charges = pd.to_numeric(df.TotalCharges, errors="coerce")  # converting to float64
logger.info(total_charges)
logger.info(type(total_charges))
logger.info(
    df[total_charges.isnull()][["customerID", "TotalCharges"]]
)  # logging null valued rows

df.TotalCharges = total_charges
mean_total_charges = df["TotalCharges"].mean()
sum_total_charges = df["TotalCharges"].sum()
logger.info(
    f"mean of TotalCharges attribute: {mean_total_charges}\nsum of TotalCharges attribute: {sum_total_charges}"
)
df.TotalCharges = df.TotalCharges.fillna(
    mean_total_charges
)  # setting null values to column mean
logger.info(df.TotalCharges)
