import argparse
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.fixes import loguniform
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, VarianceThreshold


from models.timeseries.tabular import PreProcessor
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default='tsfresh')
    parser.add_argument("--data_dir", type=str, default="dataset/cinc2020/raw/")
    parser.add_argument("--csv_dir", type=str, default="dataset/cinc2020/processed/")
    parser.add_argument("--log_dir", type=str, default="./logs/tabular/")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--cache", action='store_true', default=False)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def train(args):
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.debug:
        args.cache = False

    train_data = pd.read_csv(os.path.join(args.csv_dir, 'y_train.csv'))
    val_data = pd.read_csv(os.path.join(args.csv_dir, 'y_val.csv'))
    test_data = pd.read_csv(os.path.join(args.csv_dir, 'y_test.csv'))

    if args.debug:
        train_data = train_data.head(100)
        val_data = val_data.head(50)
        test_data = test_data.head(50)

    target_names = train_data.drop(columns=['idx']).columns.to_list()

    X_train = train_data['idx'].to_list()
    y_train = train_data.drop(columns=['idx']).to_numpy()

    X_val = val_data['idx'].to_list()
    y_val = val_data.drop(columns=['idx']).to_numpy()

    X_test = test_data['idx'].to_list()
    y_test = test_data.drop(columns=['idx']).to_numpy()

    if args.features == 'ecg':
        preprocessor = PreProcessor(data_dir=args.data_dir)

        if not args.cache:
            X_train = preprocessor.fit_transform(X_train, verbose=True)
            X_test = preprocessor.transform(X_test, verbose=True)
            X_val = preprocessor.transform(X_val, verbose=True)

            
            if not args.debug:
                os.makedirs(
                    os.path.join(args.log_dir, 'cache'),
                    exist_ok=True,
                )

                joblib.dump(X_train, os.path.join(args.log_dir, 'cache/X_train.joblib'))
                joblib.dump(X_val, os.path.join(args.log_dir, 'cache/X_val.joblib'))
                joblib.dump(X_test, os.path.join(args.log_dir, 'cache/X_test.joblib'))
        else:
            X_train = joblib.load(os.path.join(args.log_dir, 'cache/X_train.joblib'))
            X_test = joblib.load(os.path.join(args.log_dir, 'cache/X_test.joblib'))
            X_val = joblib.load(os.path.join(args.log_dir, 'cache/X_val.joblib'))
    
    elif args.features == 'tsfresh':
        X_train = pd.read_csv(os.path.join(args.csv_dir, 'train_features.csv'))
        X_val = pd.read_csv(os.path.join(args.csv_dir, 'val_features.csv'))
        X_test = pd.read_csv(os.path.join(args.csv_dir, 'test_features.csv'))


    X = np.vstack([X_train, X_val])
    y = np.vstack([y_train, y_val])
    test_fold = X_train.shape[0] * [-1] + X_val.shape[0] * [0]
    splitter = PredefinedSplit(test_fold)

    preprocessor = Pipeline([
        ('scl', StandardScaler()),
        ('missing_handler', SimpleImputer(strategy='constant', fill_value=-999)),
        # ('missing_handler', FeatureUnion(
        #     transformer_list=[
        #         ('features', SimpleImputer(strategy='constant', fill_value=-999)),
        #         ('indicators', MissingIndicator())
        #     ]
        # )),
        ('vt', VarianceThreshold(threshold=0.16)),
        ('fs', RFECV(RandomForestClassifier(n_jobs=-1, random_state=42), step=100, cv=3, scoring='f1_micro')),
    ])

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', None)
    ])

    params = [
        # {
        #     'pre__fs__min_features_to_select': np.arange(100, 501, 100),
        #     'clf': [RandomForestClassifier(random_state=args.seed, n_jobs=-1)],
        #     'clf__n_estimators': [100, 500, 1000, 2000],
        #     'clf__max_depth': np.arange(10, 100, 10),

        #     'clf__class_weight': ['balanced', None],
        #     'clf__max_features': ['sqrt', 'log2'],
        #     # 'clf__min_samples_split': [2, 10, 50],
        #     # 'clf__min_samples_leaf': [1, 5, 10],
        # },
        {
            'pre__fs__min_features_to_select': Integer(10, X.shape[1]),

            'clf': [XGBClassifier(tree_method='gpu_hist', gpu_id=0)],

            'clf__max_depth': Integer(2, 100),
            'clf__gamma': Real(1e-3, 1e+3, prior='log-uniform'),
            'clf__eta': Real(1e-3, 1e+3, prior='log-uniform'),
            'clf__scale_pos_weight': Real(1e-1, 1e+3, prior='log-uniform'),
            'clf__reg_lambda': Real(1e-3, 1e+3, prior='log-uniform'),
            'clf__reg_alpha': Real(1e-3, 1e+3, prior='log-uniform'),

            # 'clf__objective': ['binary:logistic'],
            # 'clf__min_child_weight': [1, 2, 5, 10],
            # 'clf__subsample': [0.1, 0.2, 0.3],
            # 'clf__colsample_bytree': np.linspace(0, 0.9, 10),
            
        }
    ]

    # import ipdb; ipdb.set_trace()
        
    search = BayesSearchCV(
        pipeline,
        params,
        cv=splitter,
        scoring='f1_micro',
        n_jobs=-1,
        verbose=10,
        random_state=args.seed,
        n_iter=args.n_iter,
    )
    search.fit(X, y)
    print('Best params:', search.best_params_)

    y_pred = search.predict(X_test)
    y_prob = search.predict_proba(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    np.savez_compressed(os.path.join(args.log_dir, 'test_prob'), y_prob)

    # Save report csv
    report = classification_report(y_test, y_pred, 
            target_names=target_names, zero_division=0, output_dict=True)
    report = pd.DataFrame(report).round(2).T.to_csv(
        os.path.join(args.log_dir, 'report.csv')
    )

    # Save the results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(os.path.join(args.log_dir, 'cv_results.csv'), index=False)
    joblib.dump(search, os.path.join(args.log_dir, 'search.pkl'))


if __name__ == "__main__":
    args = parse_args()
    train(args)

