import numpy as np


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


def RMSE(v, v_):
    """
    Mean squared error.
    :param v: np.array, ground truth.
    :param v_: np.array, prediction.
    :return: float, RMSE averages on all elements of input.
    """
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    """
    Mean absolute error.
    :param v: np.array, ground truth.
    :param v_: np.array, prediction.
    :return: float, MAE averages on all elements of input.
    """
    return np.mean(np.abs(v_ - v))


def MAPE(v, v_):
    """
    Mean absolute percentage error.
    :param v: np.array, ground truth.
    :param v_: np.array, prediction.
    :return: float, MAPE averages on all elements of input.
    """
    return np.mean(np.true_divide(np.abs(v_ - v), v))


def get_acc():
    pass


def RSE(v, v_):
    """
    Root Relative Squared Error.
    :param v: np.array, ground truth
    :param v_: np.array, prediction.
    :return: float, RSE on all elements of input.
    """
    return np.sqrt(np.mean((v_ - v)**2)) / normal_std(v)


def sMAPE(v, v_, c=0.01):
    """
    Symmetric Mean Absolute Percentage Error
    :param v: np.array, ground truth
    :param v_: np.array, prediction.
    :param c: float, Denominator constant
    :return: float, sMAPE on all elements of input.
    """
    # smape = np.mean(np.true_divide(np.abs(v - v_), v + v_ + c))
    # if smape < 0:
    #     print("v:", v)
    #     print("v_:", v_)
    # return smape
    if np.mean(v) < 1:
        c = 0.05
    return np.mean(np.true_divide(np.abs(v - v_), np.abs(v) + np.abs(v_) + c))


def sMAPE2(v, v_, c=0.01):
    """
    Symmetric Mean Absolute Percentage Error
    :param v: np.array, ground truth
    :param v_: np.array, prediction.
    :param c: float, Denominator constant
    :return: float, sMAPE on all elements of input.
    """
    # smape = np.mean(np.true_divide(np.abs(v - v_), v + v_ + c))
    # if smape < 0:
    #     print("v:", v)
    #     print("v_:", v_)
    # return smape
    if np.mean(v) < 1:
        c = 0.05
    return np.mean(np.true_divide(np.abs(v - v_), (np.abs(v) + np.abs(v_))/2 + c))


def sMAPE3(v, v_, c=0.01):
    """
        Symmetric Mean Absolute Percentage Error
        :param v: np.array, ground truth
        :param v_: np.array, prediction.
        :param c: float, Denominator constant
        :return: float, sMAPE on all elements of input.
        """
    # smape = np.mean(np.true_divide(np.abs(v - v_), v + v_ + c))
    # if smape < 0:
    #     print("v:", v)
    #     print("v_:", v_)
    # return smape
    Cs = np.zeros(len(v))
    for i in range(1, len(Cs)):
        y_mean = np.mean(v[0:i])
        Cs[i] = np.mean(np.abs(v[0:i]-y_mean))

    return np.mean(np.true_divide(np.abs(v - v_), (np.abs(v) + np.abs(v_)) / 2 + Cs))