import unittest
from sklearn.metrics import mean_squared_error


class TestHousingLibrary(unittest.TestCase):
    def test_mean_squared_error(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        mse = mean_squared_error(y_true, y_pred)
        self.assertEqual(mse, 0)

        y_true = [1, 2, 3]
        y_pred = [4, 5, 6]
        mse = mean_squared_error(y_true, y_pred)
        self.assertEqual(mse, 9)


if __name__ == "main":
    unittest.main()
