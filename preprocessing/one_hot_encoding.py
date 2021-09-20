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

    def __init__(self, n_min=1, load_if_exists=False,  drop_na=True, verbose=False,
                 filename="dict_nominal_columns.pickle"):

        # Instance attributes

        self.n_min = n_min
        self.load_if_exists = load_if_exists
        self.verbose = verbose
        self.drop_na = drop_na
        self.filename = filename
        self.categories = {}
        self.frequency = {}
        self.to_encode = []
        self.new_columns = []
        self.info_dict = {}

    def fit(self, data: DataFrame):

        if os.path.isfile(self.filename) & self.load_if_exists:
            pickle_in_unique = open(self.filename, "rb")
            d = pickle.load(pickle_in_unique)
            self.categories.update(d["categories"])
            self.frequency.update(d["frequency"])

        else:
            columns_to_encode = [column for column in data.columns if data[column].dtype == object]
            for column in columns_to_encode:
                # Frequency
                counts = data[column].value_counts(dropna=self.drop_na).sort_values(ascending=False)
                self.frequency.update({column: counts.to_dict()})

                # Categories
                categories = counts[counts >= self.n_min].index.tolist()
                self.categories.update({column: categories})

            pickle.dump({"categories": self.categories,
                         "frequency": self.frequency},
                        open(self.filename, 'wb'))

        self.to_encode = list(self.categories.keys())

        return self

    def transform(self, data,  max_cat=None, drop_last=False, drop_remain=False):

        """

        Parameters
        ----------
        data
        max_cat
        drop_last
        drop_remain

        Returns
        -------

        """

        self.new_columns = []

        columns_to_encode = [x for x in self.to_encode if x in data.columns]

        for count, column in enumerate(columns_to_encode, start=1):

            unique_values_column = self.categories[column]

            if drop_last:
                unique_values_column = unique_values_column[:-1]

            if max_cat is not None:
                unique_values_column = unique_values_column[:max_cat]

            msg_01 = f"Encoding: {column} ..."
            msg_02 = f"Encoding column number {count} from {len(columns_to_encode)} original columns."
            msg_03 = f"Number of unique values to encode: {len(unique_values_column)}"

            for n_val, val in enumerate(unique_values_column, start=1):

                msg_04 = f"Encoding category number {n_val} from {len(unique_values_column)}."
                name_temp = column + "_" + str(val)
                self.new_columns.append(name_temp)

                print_if(self.verbose, msg_01)
                print_if(self.verbose, msg_02)
                print_if(self.verbose, msg_03)
                print_if(self.verbose, msg_04)

                if self.verbose:
                    clear_output(wait=True)

                data[name_temp] = (data[column] == val) * 1.0

                msg_05 = f"Total of new columns created {len(self.new_columns)}."
                print_if(self.verbose, msg_05)

        if drop_remain:
            rest = [x for x in data.columns if x not in self.new_columns]
            data = data.drop(rest, axis=1)

        return data
