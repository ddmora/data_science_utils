from scipy.stats import ks_2samp


def ks_stat(y, yhat):

    """

    This function calculates the Kolmogorov KS-Statistic

    Params
    ------

    y: list-array like
       A list or an array of a  binary or continous variable.
    y_hat: list-array-like



    """
    return ks_2samp(yhat[y == 1], yhat[y != 1]).statistic