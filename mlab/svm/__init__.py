"""Support vector machines: primal linear SVM, kernel SVC (SMO) and kernel SVR."""

from mlab.svm._svc import SVC
from mlab.svm._svm import SVM
from mlab.svm._svr import SVR

__all__ = ["SVM", "SVC", "SVR"]
