import pandas as pd
import logging
logging.getLogger().setLevel(logging.INFO)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
logging.info(len(df))
# logging.info(df.head())
logging.info(df.dtypes)