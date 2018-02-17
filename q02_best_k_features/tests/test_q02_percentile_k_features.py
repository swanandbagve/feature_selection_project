from ..build import percentile_k_features
from unittest import TestCase
from inspect import getargspec
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')
expected = ['OverallQexitual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
top_features = percentile_k_features(data)

class TestPercentile_k_features(TestCase):
    def test_percentile_k_features_arguments(self):
        # Input parameters tests
        args = getargspec(percentile_k_features)
        self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (2, len(args[0])))
    def test_percentile_k_features_defaults(self):
        args = getargspec(percentile_k_features)
        self.assertEqual(args[3], (20,), "Expected default values do not match given default values")

        # Return data types
    def test_percentile_k_features_return_instance(self):
        self.assertIsInstance(top_features, list,
                              "Expected data type for return value is `List`, you are returning %s" % (
                                  type(top_features)))

        # Return values tests
    def test_percentile_k_features_return_list(self):
        self.assertEqual(top_features, expected, "Expected list of variables does not match returned list of variables")
