import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import pickle
from .utils import opts


class ClassificationToolkit:
    def __init__(self, df, y, output_dir, train_split=0.2, opts=opts):
        self.df = df
        self.ycol = y
        self.output_dir = output_dir
        self.opts = opts
        y = df[[self.ycol]]
        x_cols = [x for x in list(df.columns) if x != y]
        X = df[x_cols]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=train_split, random_state=1, stratify=y)
        undersample = RandomUnderSampler()
        self.under_x_train, self.under_y_train = undersample.fit_resample(self.X_train.to_numpy(), self.y_train)
        oversample = RandomOverSampler()
        self.over_x_train, self.over_y_train = oversample.fit_resample(self.X_train.to_numpy(), self.y_train)
        smote = SMOTE()
        self.smote_x_train, self.smote_y_train = smote.fit_resample(self.X_train.to_numpy(), self.y_train)


    def run_classifications(self):
        sampling = ['unsampled', 'undersampled', 'oversampled', 'smote']
        for cls in self.opts:
            model = cls['model']
            evals = cls['evals']
            search_space = cls['search_space']
            name = cls['name']
            for samp in sampling:
                if samp == "undersampled":
                    x = self.under_x_train
                    y = self.under_y_train
                elif samp == "oversampled":
                    x = self.over_x_train
                    y = self.over_y_train
                elif samp == "smote":
                    x = self.smote_x_train
                    y = self.smote_y_train
                else:
                    x = self.X_train
                    y = self.y_train
                self.run_classifier(x, y, self.X_test, self.y_test, model, search_space, evals, f"{name}_{samp}")


    def run_classifier(self, x_train, y_train, x_test, y_test, method, search_space, max_evals, prefix):

        def method_inner(search_space):
            model = method(**search_space)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            f1 = f1_score(y_test, y_pred, average='micro')
            return {'loss': -f1, 'status': STATUS_OK, 'model': model, 'pred': y_pred}

        trials = Trials()
        best_params = fmin(fn=method_inner, space=search_space, max_evals=max_evals, trials=trials, algo=tpe.suggest)
        space_eval(search_space, best_params)
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        pickle.dump(best_model, open(os.path.join(self.output_dir, f'{prefix}_model.sav'), 'wb'))
        best_predicted_results = trials.results[np.argmin([r['loss'] for r in trials.results])]['pred']
        report = pd.DataFrame(classification_report(y_test, best_predicted_results, output_dict=True)).transpose()
        report.to_csv(os.path.join(self.output_dir, f'{prefix}_classification_report.csv'), index=False)











