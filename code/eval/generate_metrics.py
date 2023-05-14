import json
import glob
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder

from eval import supervised_model_training, get_utility_metrics, stat_sim


def generate_metrics(data_path):
    model_types =  ['ae', 'dae', 'ee', 'vae']
    real_path = data_path + 'real.csv'
    info_path = data_path + 'info.json'
    res = pd.DataFrame(columns=["model_type", "wd", "jsd", "corr_diff", "acc", "auc", "f1"])

    real = pd.read_csv(real_path)
    target_col = real.columns[-1]

    with open(info_path, 'r') as file:
        info_json = file.read()
    info = json.loads(info_json)

    if info['target_encode'] == "True":
        le = LabelEncoder()
        le.fit(real[target_col])
        real[target_col] = le.transform(real[target_col])

    for model_type in model_types:
        real_eval = real.copy()
        fake_paths = glob.glob(data_path + model_type + '/*_[0-9]*.csv')
        
        # read data
        fake = [pd.read_csv(f) for f in fake_paths]
        
        if info['target_encode'] == "True":
            for df in fake: 
                df[target_col] = le.transform(df[target_col])
            
        # encode categorical cols
        if info['encode_required']:
            enc = pd.concat([real_eval, pd.concat(fake)])
            enc = pd.get_dummies(enc, columns=info['encode_required'])

            # reorder
            order = list(enc.columns)
            order.remove(target_col)
            order.append(target_col)
            enc = enc[order]

            real_eval = enc.iloc[:real.shape[0]]
            prev = real.shape[0]
            fake_enc = []

            for df in fake:
                fake_enc.append(enc.iloc[prev:(prev + df.shape[0])])
                prev += df.shape[0]

            fake = fake_enc

        # statistical similarity
        stat = [stat_sim(real_path, f, list(info['discrete_cols']), eval(info['target_encode'])) for f in fake_paths]
        wd, jsd, corr_diff = np.array(stat).mean(axis=0)

        # ml utility
        ml_diff = get_utility_metrics(real_eval, fake, test_ratio=info['test_size'])
        acc, auc, f1 = ml_diff.mean(axis=0)

        res = res.append({"model_type": model_type, "wd": wd, "jsd": jsd, "corr_diff": corr_diff, "acc": acc, "auc": auc, "f1": f1}, ignore_index=True)

    res.to_csv(data_path + "result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "--data", help="root path contatining all data folders")
    args = parser.parse_args()

    datasets = ["intrusion"]

    for data in datasets:
        print(f"processing: {data}")
        generate_metrics(args.d + f"{data}/")





