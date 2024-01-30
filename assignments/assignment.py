# %%
import numpy as np
import pandas as pd
from typing_extensions import Any, Self


# TODO: Implement the LogisticRegression class
class LogisticRegression:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self._errors = []
        self.w_ = None
        self.b_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)+ self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Samples.

        Returns
        -------
        C : array, shape = [n_examples]
            Predicted class label (1 or 0) per sample.

        # TODO: Implement this function to predict class labels for samples in X.
        # Use the weights and bias unit from the fitting process.
        # You may find the predict_proba method useful.
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Samples.

        Returns
        -------
        P : array, shape = [n_examples, 2]
            The class probabilities of the input samples. The order of the classes corresponds to that in the attribute `classes_`.

        # TODO: Implement this function to predict class probabilities for samples in X.
        # Use the weights and bias unit from the fitting process.
        # You will implement the sigmoid function
        """
        prob = 1/(1+ np.exp(net_input))
        return np.array([prob, 1-prob])

    def copy(self):
        model = LogisticRegression(
            eta=self.eta,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], np.ndarray):
                model.__dict__[attr] = self.__dict__[attr].copy()
            else:
                model.__dict__[attr] = self.__dict__[attr]
        return model


# TODO: Implement the logistic regression with L2 regularization
class LogisticRegressionRidge(LogisticRegression):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    gamma: float
      Regularization parameter
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """

    def __init__(
        self, eta: float = 0.01, n_iter: int = 50, gamma: float = 0, random_state=1
    ) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.gamma = gamma
        self._errors = []
        self.w_ = None
        self.b_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)

        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                # the l2 part
                self.w_ += update * xi - self.eta * self.gamma * self.w_
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        

class OneVsRest:
    """
    OneVsRest class for multi-class classification.
    This class uses the one-vs-rest (OvR) method for multi-class classification.
    """

    def __init__(self, classifier: LogisticRegression, n_classes: int):
        """
        Initialize the OneVsRest class.

        Parameters
        ----------
        classifier : object
            The classifier to be used for the one-vs-rest classification.
            The classifier must have a .fit, .predict, .copy method.
        n_classes : int
            The number of classes in the target variable.
        """
        self.n_classes = n_classes
        self.classifiers = []
        for i in range(self.n_classes):
            self.classifiers.append(classifier.copy())

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        # Hint
        Step 1: Iterate over each class in a one-vs-rest manner.
        Step 2: For each class, create a binary target vector where the current class is 1 and all others are 0.
        Step 3: Use the classifier's .fit method to train the classifier on the training data and the binary target vector.
        Step 4: Store the trained classifier for later use in prediction.
        """
        for i in range(self.n_classes):
          targetvector=np.where(y==i, 1,0)
          self.classifiers[i].fit(X, targetvector)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_examples]
            Predicted class label per sample.

        # Hint
        Step 1: Iterate over each classifier.
        Step 2: Use the classifier's .predict_proba method to get the class probabilities.
        Step 3: Store the class probabilities in a matrix.
        Step 4: Return the class with the highest probability for each sample.
        """
        classprob=()
        for classifier in self.classifiers:
            classprob= classprob.append(classifier.predict_proba(X))
        return np.argmax(classprob, axis=1)


# %%
