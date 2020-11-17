
import collections
import numpy as np
from sklearn.model_selection import train_test_split

def balance_data(X, y):
    c = collections.Counter(y)
    mini = min(c.values())
    print(mini)
    d = collections.defaultdict(lambda:0)
    outX, outy = list(), list()
    for inX, iny in zip(X, y):
        if(d[iny]<mini):
            outX.append(inX)
            outy.append(iny)
            d[iny]+=1
    return np.asarray(outX), np.asarray(outy)

def data_split(X, y, ratio, **kwargs):
    """Split a dataset (X, inputs), (y, outputs) following the ratio parameter with a shuffle
    it uses the train_test_split function which can receive parameters
        (but train_size and test_size) through **kwargs
    """
    if(not isinstance(ratio, collections.abc.Iterable)):
        ratio=[ratio]
    else:
        ratio=list(ratio)
    assert len(ratio)>=1
    assert "train_size" not in kwargs
    assert "test_size" not in kwargs
    while sum(ratio) >= 1:
        ratio.pop()
    if(len(ratio)==1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio[0], **kwargs)
        return [(np.asarray(X_train), np.asarray(y_train))] + [(np.asarray(X_test), np.array(y_test))]
    else:
        current_ratio = ratio.pop(0)
        ratio = [p/(1-current_ratio) for p in ratio]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=current_ratio, **kwargs)
        return [(np.asarray(X_train), np.asarray(y_train))] + data_split(X_test, y_test, ratio, **kwargs)
