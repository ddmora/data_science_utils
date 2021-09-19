import os
import pickle
from IPython.display import clear_output
from pandas import DataFrame


def print_if(boolean, message, *kwargs):
    if boolean:
        print(message, *kwargs)


class OneHotEncoder:

    """

    Parameters
    ----------
    max_cat: int, default=None
        The maximum number of categories per column
    n_min: int, default=2
        The minimum number of observations to be classified as a category
    load_if_exists: bool, default=False
        If True, Loads the dictionary of categories if is already in the ''. This functionality speed up the time of
        encoding, by loading the existing dictionary of categories. Mainly useful when handling large datasets.
    drop_na: bool, default=True
        If True, drops the missing .
    verbose: bool, default False
        If True, print the report of encoding process.

    Attributes
    ----------
    categories: dict
        A dictionary with column names as entries, and the respective categories, determined during fitting process.
    to_encode: list
        List of columns encoded.
    new_columns: list
        List of names of the new columns created by the one hot encoding
    info_dict: dict
        A dictionary with column names as entries, and the respective observations during the process if exist.


    """

    # Class attributes

    def __init__(self, max_cat=None, n_min=1, load_if_exists=False, drop_last=True, drop_na=True, verbose=False,
                 filename="dict_nominal_columns.pickle"):

        # Instance attributes

        self.max_cat = max_cat
        self.n_min = n_min
        self.load_if_exists = load_if_exists
        self.verbose = verbose
        self.drop_last = drop_last
        self.drop_na = drop_na
        self.filename = filename
        self.categories = {}
        self.to_encode = []
        self.new_columns = []
        self.info_dict = {}

    def fit(self, data: DataFrame):

        if os.path.isfile(self.filename) & self.load_if_exists:
            pickle_in_unique = open(self.filename, "rb")
            self.categories.update(pickle.load(pickle_in_unique))

        else:
            for column in data.columns:
                if data[column].dtype == object:
                    counts = data[column].value_counts(dropna=self.drop_na).sort_values(ascending=False)
                    counts = counts[counts >= self.n_min]
                    categories = counts.index.tolist()

                    if self.max_cat is None:
                        self.categories.update({column: categories})

                    else:
                        self.categories.update({column: categories[:self.max_cat]})
                        msg = f"{self.max_cat} selected from {len(categories)} categories."
                        self.info_dict.update({column: msg})

            pickle.dump(self.info_dict, open(self.filename, 'wb'))

        self.to_encode = list(self.categories.keys())

        return self

    def transform(self, data, drop_remain=True):

        for count, column in enumerate(self.to_encode, start=1):
            if column in data.columns:
                unique_values_column = self.categories[column]

                if self.drop_last & (column not in self.info_dict):
                    unique_values_column = unique_values_column[:-1]

                msg_01 = f"Encoding: {column} ..."
                msg_02 = f"Encoding column number {count} from {len(self.to_encode)} original columns."
                msg_03 = f"Number of unique values: {len(unique_values_column)}"

                for n_val, val in enumerate(unique_values_column, start=1):

                    msg_04 = f"Encoding category number {n_val} from {len(unique_values_column)}."
                    name_temp = column + "_" + str(val)
                    self.new_columns.append(name_temp)
                    msg_05 = f"Total of new columns created {len(self.new_columns)}."

                    print_if(self.verbose, msg_01)
                    print_if(self.verbose, msg_02)
                    print_if(self.verbose, msg_03)
                    print_if(self.verbose, msg_04)
                    print_if(self.verbose, msg_05)

                    if self.verbose:
                        clear_output(wait=True)

                    data[name_temp] = (data[column] == val) * 1.0

                msg_05 = f"Total of new columns created {len(self.new_columns)}."
                print_if(self.verbose, msg_05)

        if drop_remain:
            rest = [x for x in data.columns if x not in self.new_columns]
            data = data.drop(rest, axis=1)

        return data
