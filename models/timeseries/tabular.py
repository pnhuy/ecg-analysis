import os
from sklearn.base import BaseEstimator, TransformerMixin
from models.timeseries.ecg_features.feature_extractor import Features

class PreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir=''):
        self.data_dir = data_dir
        self.columns = None

    def extract_features(self, file_names, verbose=False):
        self.extractor = Features(
            file_names=file_names,
            feature_groups=['full_waveform_features']
        )
        self.extractor.extract_features(
            filter_bandwidth=[3, 45], n_signals=None, show=verbose, 
            labels=None, normalize=True, polarity_check=True,
            template_before=0.25, template_after=0.4
        )
        features = self.extractor.get_features().drop(columns=['file_name'])
        return features
    
    def fit(self, X, y=None, verbose=False):
        self.verbose = verbose
        return self

    def transform(self, X, y=None, verbose=None):
        if self.data_dir:
            file_names = [os.path.join(self.data_dir, x) for x in X]
            file_names = [f"{fp}.hea" if not fp.endswith('.hea') else fp for fp in file_names ]
        else:
            file_names = X
        ft = self.extract_features(file_names, verbose=self.verbose or verbose)
        
        if not self.columns:
            self.columns = ft.columns.to_list()

        ft = ft.to_numpy()

        return ft