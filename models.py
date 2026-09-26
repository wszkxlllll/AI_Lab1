"""四种分类器的构造函数。"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def make_model(name, params, seed):
    if name == "NB":
        return MultinomialNB(**params)
    if name == "SVM":
        return SVC(kernel="linear", cache_size=512, random_state=seed, **params)
    if name == "LR":
        return LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed, **params)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden"]), alpha=params["alpha"],
            activation="relu", solver="adam", batch_size=256, learning_rate_init=0.001,
            max_iter=300, tol=1e-4, n_iter_no_change=15, early_stopping=False,
            random_state=seed,
        )
    raise ValueError(name)


def specification(model, params, representation="tfidf", view="joined", title_weight=1.0):
    return dict(model=model, params=params, representation=representation, view=view, title_weight=title_weight)
