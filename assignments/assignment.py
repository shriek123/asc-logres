# %%
import numpy as np
import pandas as pd
from typing import Any, Self


# TODO: Implement the Perceptron class
class Perceptron:
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
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        pass


# TODO: Implement the Adaline class
class Adaline(Perceptron):
    def __init__(self, eta: float, n_iter: int) -> None:
        super(Adaline, self).__init__(eta=eta, n_iter=n_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass

    def _prediction_linear(self, X: np.ndarray) -> np.ndarray:
        pass


# TODO: Implement the function to load the penguin data.
def load_penguin_data():
    """
    This function performs the following steps:
    1. Loads the penguin dataset from the specified URL (provided in the function).
    2. Drops any rows with missing values using the `dropna` function from pandas.
    3. Creates a pandas dataframe by selecting the 'bill_length_mm' and 'bill_depth_mm' columns.
    4. Converts the selected dataframe to a numpy array, denoted as 'X'.
    5. Slices the 'species' column from the original dataframe.
    6. Converts the 'species' column to a numpy array, denoted as 'labels'.
    7. Returns the numpy arrays 'X' and 'labels' in that order.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays. The first array, 'X', contains the bill length and depth measurements. The second array, 'labels', contains the species information.
    """
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv"
    pass


# %%
