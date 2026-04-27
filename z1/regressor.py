#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Алекса Ћетковић <cetkovic.sv77.2022@uns.ac.rs>
# Code that compares different approaches can be found at: 
# https://github.com/cetkovicaleksa/ml-assignments/tree/master/z1/regression.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import (
    make_column_transformer,
    TransformedTargetRegressor
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error

from sklearn import set_config
set_config(transform_output="pandas")

seed = 14_02_2003
np.random.seed(seed)


def read_json(path):
    df = pd.read_json(path)

    df['Sprat'] = (
        df['Sprat'].replace({
            "potkrovlje": -2,
            "suteren": -1,
            "nisko prizemlje": -0.5,
            "prizemlje": 0,
        })
    )
    
    df[['Slike', 'Cena', 'Kvadratura', 'Sobe', 'Sprat']] = (
        df[['Slike', 'Cena', 'Kvadratura', 'Sobe', 'Sprat']]
        .apply(pd.to_numeric, errors="coerce")
    )

    df[['Grad', 'Naziv', 'Prodavac', 'Uknjizen', 'Garaza', 'Parking']] = (
        df[['Grad', 'Naziv', 'Prodavac', 'Uknjizen', 'Garaza', 'Parking']]
        .astype("category")
    )

    return df.convert_dtypes()

train_json = Path(sys.argv[1])
df = read_json(train_json)

X = df.drop('Cena', axis=1)
y = df['Cena']

y_transform = FunctionTransformer(np.log1p, np.expm1, feature_names_out="one-to-one")

cat_preprocessor = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
)

num_preprocessor = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

kridge_pre = make_column_transformer(
    (cat_preprocessor, ['Grad', 'Prodavac', 'Garaza', 'Uknjizen', 'Parking']),
    (num_preprocessor, ['Kvadratura', 'Slike', 'Sobe', 'Sprat']),
)

kridge = Pipeline([
    ("pre", kridge_pre),
    ("model", TransformedTargetRegressor(
        regressor=KernelRidge(),
        transformer=y_transform,
    ))
])

kridge_grid = {
    "model__regressor__alpha": [0.1, 1, 10, 100],
    "model__regressor__kernel": ["rbf", "linear"],
    "model__regressor__gamma": [0.001, 0.01, 0.1],
}

gcv_kridge = GridSearchCV(
    estimator=kridge,
    param_grid=kridge_grid,
    cv=5,
    scoring="neg_mean_absolute_percentage_error",
    return_train_score=True,
    n_jobs=-1,
    verbose=0
)

gcv_kridge.fit(X, y)

estimator = gcv_kridge.best_estimator_

test_json = Path(sys.argv[2])
df_test = read_json(test_json)

X_test = df_test.drop('Cena', axis=1)
y_test = df_test['Cena']

y_pred = estimator.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(mape)
