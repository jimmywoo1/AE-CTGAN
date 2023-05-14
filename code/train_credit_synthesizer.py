import datetime
import pandas as pd
from ctgan.synthesizers import CTGAN, TVAE

epochs = [30, 50, 75, 100, 200]
synthesizers = ["ctgan", "tvae"]
credit_df = pd.read_csv(f"../dataset/credit/creditcard.csv")

for synthesizer in synthesizers:
    for num_epoch in epochs:
        print(f"Training {synthesizer} for {num_epoch} epochs")
        if synthesizer == "ctgan":
            model = CTGAN(epochs=num_epoch)
        elif synthesizer == "tvae":
            model = TVAE(epochs=num_epoch)
        model.fit(credit_df, discrete_columns=["Class"])
        now = datetime.datetime.now()
        current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
        model.save(f"../models/credit_{synthesizer}_{num_epoch}_epochs_{current_time}.pkl")