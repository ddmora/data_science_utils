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
        The maximun number of categories per column
    n_min: int, default=2
        The minimun number of observations to be classified as a category
    load_if_exists: bool, default=False
        If True, Loads the dictionary of categories if is already in the ''. This functionality speed up the time of
        encoding, by loading the existing dictionary of categories. Mainly useful when handling large datasets.
    drop: bool, default=True
        If True, drops the original columns.
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
        A dictionary with column names as entries, and the respecitive observations during the process if exist.


    """

    # Class attributes

    def __init__(self, max_cat=None, n_min=2, load_if_exists=False, drop=True, drop_last=True, verbose=False,
                 filename="dict_nominal_columns.pickle"):

        # Instance attributes

        self.max_cat = max_cat
        self.n_min = n_min
        self.load_if_exists = load_if_exists
        self.drop = drop
        self.verbose = verbose
        self.drop_last = drop_last
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
                    counts = data[column].value_counts().sort_values(ascending=False)
                    counts = counts[counts >= self.n_min]
                    categories = [str(x) for x in counts.index]

                    if self.max_cat is None:
                        self.categories.update({column: categories})

                        self.categories.update({column: categories[:self.max_cat]})
                        self.info_dict.update({column: 'exceeds the maximum limit of categories'})
            pickle.dump(self.info_dict, open(self.filename, 'wb'))

        self.to_encode = list(self.categories.keys())

        return self

    def transform(self, data):

        count = 0
        for column in self.to_encode:
            if column in data.columns:

                unique_values_column = self.categories[column]
                # We delete the dummy associated with None if present in unique values.
                if "nan" in unique_values_column:
                    unique_values_column = [x for x in unique_values_column if x != "nan"]
                # ELse we omitted one level
                if self.drop_last & (column not in self.info_dict):
                    unique_values_column = unique_values_column[:-1]

                count += 1
                to = len(self.to_encode)
                n_cat = len(unique_values_column)

                msg_01 = "Encoding: {} ...".format(column)
                msg_02 = "Encoding column number {} from {} original columns.".format(count, to)
                msg_03 = "Number of unique values: {}".format(n_cat)

                count_2 = 0

                for val in unique_values_column:

                    count_2 += 1

                    print_if(self.verbose, msg_01)
                    print_if(self.verbose, msg_02)
                    print_if(self.verbose, msg_03)

                    msg_04 = "Encoding category number {} from {}".format(count_2, n_cat)
                    print_if(self.verbose, msg_04)

                    name_temp = column + "_" + str(val)
                    self.new_columns.append(name_temp)
                    new = len(self.new_columns)
                    msg_05 = "Total of new columns created {}.".format(new)
                    print_if(self.verbose, msg_05)

                    data[name_temp] = (data[column] == str(val)) * 1.0

                    if self.verbose:
                        clear_output(wait=True)

        if self.drop:
            data = data.drop(self.to_encode, axis=1)

        return data
