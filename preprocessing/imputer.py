from pandas import DataFrame


class BasicImputer:
    def __init__(self, imputed_values_dict, to_impute):
        self.imputed_values_dict = imputed_values_dict
        self.to_impute = to_impute

    def fit_transform(self, data: DataFrame):
        data[self.to_impute] = data[self.to_impute].fillna(self.imputed_values_dict)
        return data
