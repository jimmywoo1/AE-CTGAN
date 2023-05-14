import pandas as pd

from ctgan.synthesizers.ae_gan import CTGANV2
from ctgan.data_transformer import DataTransformer

def load_df(path: str, columns: list=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = columns
    return df

# load adult
path = "~/datasets/adult/adult.data"
cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
        "hours-per-week", "native-country", "income_50k"]
df = load_df(path, cols)

# transform data - MSN + OH
discrete_cols = ["workclass", "education", "marital-status", "occupation", "relationship",
                 "race", "sex", "native-country", "income_50k"]
dt = DataTransformer()
dt.fit(df, discrete_columns=discrete_cols)
target_index = df.shape[1] - 1
sample_length = df.shape[0]

ae_gan = CTGANV2(verbose=True)
ae_gan.fit(df, discrete_cols, dt=dt, target_index=target_index)
synth = ae_gan.sample(sample_length)



