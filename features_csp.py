import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from config import N_CSP_COMPONENTS


def make_csp_lda_ovo():
    csp  = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    return Pipeline([("csp", csp), ("ovo", OneVsOneClassifier(LDA()))])


def fit_predict(X_train, y_train, X_test, y_test):
    model = make_csp_lda_ovo()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"accuracy": float(np.mean(y_pred == y_test)),
            "y_pred": y_pred, "y_true": y_test}
