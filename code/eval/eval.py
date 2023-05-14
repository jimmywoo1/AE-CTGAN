import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train, x_test, y_test, model_name):
  
  """
  Trains and evaluates commonly used ML models
  Inputs:
  1) x_train -> indepedent features of training dataset
  2) y_train -> dependent feature of training dataset
  3) x_test -> independent features of testing dataset
  4) y_test -> dependent feature of testing dataset 
  5) model_name -> name of ML model to be used
  Outputs:
  1) List of metrics containing accuracy, auc and f1-score of trained ML model as achieved on test-set.
  
  """
  
  # Selecting the model
  if model_name == 'lr':
    model  = LogisticRegression(random_state=42,max_iter=500) 
  elif model_name == 'svm':
    model  = svm.SVC(random_state=42,probability=True)
  elif model_name == 'dt':
    model  = tree.DecisionTreeClassifier(random_state=42)
  elif model_name == 'rf':      
    model = RandomForestClassifier(random_state=42)
  elif model_name == "mlp":
    model = MLPClassifier(random_state=42,max_iter=100)

  # Fitting the model and computing predictions on test data
  model.fit(x_train, y_train)
  pred = model.predict(x_test)

  # In case of multi-class classification AUC and F1-scores are computed using weighted averages across all distinct labels
  if len(np.unique(y_test))>2:
    predict = model.predict_proba(x_test)      

    if predict.shape[1] != len(np.unique(y_test)):
      vals = np.unique(y_test)
      avail = {k: v for k, v in enumerate(np.unique(y_train))}
      missing = [x for x in vals if x not in np.unique(y_train)]

      idx = predict.shape[1]

      for m in missing:
        predict = np.column_stack((predict, np.zeros(predict.shape[0])))
        avail[idx] = m
        idx += 1
      
      idx_order = sorted(avail, key=lambda x: avail[x])
      predict = predict[:, idx_order]

    acc = metrics.accuracy_score(y_test,pred)*100
    auc = metrics.roc_auc_score(y_test, predict,average="weighted",multi_class="ovr")
    f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="weighted")[2]
    return [acc, auc, f1_score] 

  else:
    predict = model.predict_proba(x_test)[:,1]    
    acc = metrics.accuracy_score(y_test,pred)*100
    auc = metrics.roc_auc_score(y_test, predict)
    f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
    return [acc, auc, f1_score] 

def get_utility_metrics(data_real, data_fake_list, scaler="MinMax", classifiers=["lr","dt","rf","mlp"], test_ratio=.20):

    """
    Returns ML utility metrics
    Inputs:
    1) data_real ->  real dataset
    2) data_fake_list -> list of synthetic datasets
    3) scaler ->  choice of scaling method to normalize/standardize data before fitting ML model
    4) classifiers -> list of classifiers to be used
    5) test_ratio -> ratio of the size of test to train data 
    Outputs:
    1) diff_results -> matrix of average differences in accuracy, auc and f1-scores achieved on test dataset 
    between ML models trained on real vs synthetic datasets. 
    
    Note that the average is computed across the number of replications chosen for the experiment
    """
    data_real = data_real.to_numpy()

    # Spliting the real data into train and test datasets
    data_dim = data_real.shape[1]
    data_real_y = data_real[:,-1]
    data_real_X = data_real[:,:data_dim-1]

    X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio, stratify=data_real_y,random_state=42) 

    # Selecting scaling method
    if scaler=="MinMax":
      scaler_real = MinMaxScaler()
    else:
      scaler_real = StandardScaler()

    # Scaling the independent features of train and test datasets   
    scaler_real.fit(X_train_real)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    # Computing metrics across ML models trained using real training data on real test data
    all_real_results = []
    for classifier in classifiers:
      real_results = supervised_model_training(X_train_real_scaled,y_train_real,X_test_real_scaled,y_test_real,classifier)
      all_real_results.append(real_results)
      
    # Computing metrics across ML models trained using corresponding synthetic training datasets on real test data  
    all_fake_results_avg = []
    
    for i, data_fake in enumerate(data_fake_list):
      data_fake = data_fake.to_numpy()

      # Spliting synthetic data to obtain corresponding synthetic training dataset
      data_fake_y = data_fake[:,-1]
      data_fake_X = data_fake[:,:data_dim-1]
      
      # Selecting scaling method
      if scaler=="MinMax":
        scaler_fake = MinMaxScaler()
      else:
        scaler_fake = StandardScaler()
      
      # Scaling synthetic training data
      scaler_fake.fit(data_fake_X)
      X_train_fake_scaled = scaler_fake.transform(data_fake_X)
      
      # Computing metrics across ML models trained on synthetic training data on real test data
      all_fake_results = []
      for classifier in classifiers:
        fake_results = supervised_model_training(X_train_fake_scaled,data_fake_y,X_test_real_scaled,y_test_real,classifier)
        all_fake_results.append(fake_results)

      # Storing the results across synthetic datasets 
      all_fake_results_avg.append(all_fake_results)
    
    # Returning the final avg difference between metrics of ML models trained using real vs synthetic datasets. 
    diff_results = np.array(all_real_results)- np.array(all_fake_results_avg).mean(axis=0)
    return diff_results

def stat_sim(real_path,fake_path,cat_cols=None, target_encode=False):
    
  """
  Returns statistical similarity metrics
  Inputs:
  1) real_path -> path to real data
  2) fake_path -> path to synthetic data
  3) cat_cols -> list of categorical column names
    
  Outputs:
  1) List containing the difference in avg (normalized) wasserstein distance across numeric columns, avg jensen shannon divergence 
  across categorical columns and euclidean norm of the difference in pair-wise correlations between real and synthetic datasets
    
  """
    
  # Loading real and synthetic data
  real = pd.read_csv(real_path)
  fake = pd.read_csv(fake_path)

  if target_encode:
    cat_cols.append(real.columns[-1])

  # Computing the real and synthetic pair-wise correlations
  real_corr = associations(real, nominal_columns=cat_cols, compute_only=True)['corr']

  fake_corr = associations(fake, nominal_columns=cat_cols, compute_only=True)['corr']

  # Computing the squared norm of the difference between real and synthetic pair-wise correlations
  corr_dist = np.linalg.norm(real_corr - fake_corr)
    
  # Lists to store the results of statistical similarities for categorical and numeric columns respectively
  cat_stat = []
  num_stat = []
    
  for column in real.columns:
        
    if column in cat_cols:

      # Computing the real and synthetic probabibility mass distributions (pmf) for each categorical column
      real_pmf=(real[column].value_counts()/real[column].value_counts().sum())
      fake_pmf=(fake[column].value_counts()/fake[column].value_counts().sum())
      categories = (fake[column].value_counts()/fake[column].value_counts().sum()).keys().tolist()
            
      # Ensuring the pmfs of real and synthetic data have the categories within a column in the same order
      sorted_categories = sorted(categories)
            
      real_pmf_ordered = [] 
      fake_pmf_ordered = []

      for i in sorted_categories:
        real_pmf_ordered.append(real_pmf[i])
        fake_pmf_ordered.append(fake_pmf[i])
            
      # If a category of a column is not generated in the synthetic dataset, pmf of zero is assigned
      if len(real_pmf)!=len(fake_pmf):
        zero_cats = set(real[column].value_counts().keys())-set(fake[column].value_counts().keys())
        for z in zero_cats:
          real_pmf_ordered.append(real_pmf[z])
          fake_pmf_ordered.append(0)

      # Computing the statistical similarity between real and synthetic pmfs 
      cat_stat.append(distance.jensenshannon(real_pmf_ordered,fake_pmf_ordered, 2.0))        
        
    else:
      # Scaling the real and synthetic numerical column values between 0 and 1 to obtained normalized statistical similarity
      scaler = MinMaxScaler()
      scaler.fit(real[column].values.reshape(-1,1))
      l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
      l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            
      # Computing the statistical similarity between scaled real and synthetic numerical distributions 
      num_stat.append(wasserstein_distance(l1,l2))

  return [np.mean(num_stat),np.mean(cat_stat),corr_dist]