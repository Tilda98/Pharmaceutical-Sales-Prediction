import os
import sys
import unittest
import pandas as pd
import pandas.api.types as ptypes
from pandas.api import types

sys.path.append('..')
from scripts.data_cleaning import DataCleaner

class TestDataCleaner(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.cleaner = DataCleaner()
