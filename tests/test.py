import unittest
import numpy as np
import sys

sys.path.append("assignments/")
from assignment import *
import os


class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.p = Perceptron(eta=1e-2, n_iter=100)
        # Generate 10 linearly separable points
        self.X = np.array([[i, i + 1] for i in range(1, 11)])
        self.y = np.array([1 if i <= 5 else 0 for i in range(1, 11)])

    def test_fit(self):
        self.p.fit(self.X, self.y)

    def test_predict(self):
        # Run fit
        self.test_fit()

        # Prediction
        prediction = self.p.predict([[-1, -1], [20, 20]])
        self.assertEqual(len(prediction), 2)
        self.assertAlmostEqual(prediction[0], 1)
        self.assertAlmostEqual(prediction[1], 0)


class TestAdaline(unittest.TestCase):
    def setUp(self):
        self.p = Adaline(eta=1e-5, n_iter=1000)
        # Generate 10 linearly separable points
        self.X = np.array([[i, i + 1] for i in range(1, 11)])
        self.y = np.array([1 if i <= 5 else 0 for i in range(1, 11)])

    def test_fit(self):
        self.p.fit(self.X, self.y)

    def test_predict(self):
        # Run fit
        self.test_fit()

        # Prediction
        prediction = self.p.predict([[-1, -1], [20, 20]])
        self.assertEqual(len(prediction), 2)


class TestData(unittest.TestCase):
    def test_load_penguin_data(self):
        X, y = load_penguin_data()
        self.assertEqual(X.shape, (333, 2))
        self.assertEqual(y.shape, (333,))


class TestFigure(unittest.TestCase):
    def test_fig(self):
        self.assertTrue(os.path.exists("./figs/decision_region.png"))


if __name__ == "__main__":
    unittest.main()
