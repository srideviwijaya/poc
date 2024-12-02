import csv
import numpy as np
import os.path
import pandas as pd
import time
import xgboost as xgb
import sys

dmatrix_train_filename = "higgs_train.dmatrix"
dmatrix_test_filename = "higgs_test.dmatrix"
csv_filename = "HIGGS.csv.gz"
train_rows = 10500000
test_rows = 500000
num_round = 10000

def load_higgs():

    df_higgs_train = pd.read_csv(csv_filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)
    dtrain = xgb.DMatrix(df_higgs_train.iloc[:, 1:29], df_higgs_train[0])
    dtrain.save_binary(dmatrix_train_filename)
    df_higgs_test = pd.read_csv(csv_filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)
    dtest = xgb.DMatrix(df_higgs_test.iloc[:, 1:29], df_higgs_test[0])
    dtest.save_binary(dmatrix_test_filename)

    return dtrain, dtest


# dtrain, dtest = load_higgs()
dtrain = xgb.DMatrix("higgs_train.dmatrix")
dtest = xgb.DMatrix("higgs_test.dmatrix")
param = {}
param['objective'] = 'binary:logitraw'
param['eval_metric'] = 'error'
# param['tree_method'] = 'gpu_hist'
param['device'] = 'cuda'
param['silent'] = 1
param['max_depth'] = 15

print("Training with GPU ...")
tmp = time.time()
gpu_res = {}
model = xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
          evals_result=gpu_res)
gpu_time = time.time() - tmp
print("GPU Training Time: %s seconds" % (str(gpu_time)))

model.save_model("xgboost_model.bin")