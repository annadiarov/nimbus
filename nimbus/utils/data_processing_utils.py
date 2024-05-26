import numpy as np
import pandas as pd
from nimbus.utils import LoggerFactory
from nimbus.globals import LOGGER_LEVEL

logger = LoggerFactory.get_logger(__name__, LOGGER_LEVEL)


def balance_binary_data(data: pd.DataFrame,
                        label_col: str = 'label',
                        seed: int = 42):
    """
    Balance the data by undersampling the majority class to match the size of
    the minority class.
    :param data: pd.DataFrame
        Data to be balanced. Must contain a column with the labels 0 and 1.
    :param label_col: str,
        Name of the label column
    :param seed: int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    labels = data[label_col].to_numpy()

    # Get the indices of positive and negative samples
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    # Undersample the negative class to match the size of the minority class
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_samples = min(n_pos, n_neg)

    logger.info(f"There are {n_pos} positive samples and {n_neg} negative "
                f"samples. We will balance the data to {n_samples} samples "
                f"per class.")

    pos_idx = np.random.choice(pos_idx, n_samples, replace=False)
    neg_idx = np.random.choice(neg_idx, n_samples, replace=False)

    # Combine the positive and negative indices
    balanced_data = pd.concat([data.iloc[pos_idx], data.iloc[neg_idx]]).reset_index(drop=True)

    return balanced_data
