import os
import pandas as pd


def rename_test_identity_columns(df):
    """
    Correct test identity columns
    :param df:
    :return df:
    """
    for col in df.columns:
        if col[0:2] == 'id':
            new_col_name = col.split('-')[0] + '_' + col.split('-')[1]
            df = df.rename(columns={col: new_col_name})

    return df


def load_data(DATA_DIRECTORY):
    """
    load training, testing and sample submission files from a given directory
    :param DATA_DIRECTORY:
    :return dataframes:
    """
    train_identity = pd.read_csv(os.path.join(DATA_DIRECTORY, 'train_identity.csv'), index_col='TransactionID')
    train_transaction = pd.read_csv(os.path.join(DATA_DIRECTORY, 'train_transaction.csv'), index_col='TransactionID')

    test_identity = pd.read_csv(os.path.join(DATA_DIRECTORY, 'test_identity.csv'), index_col='TransactionID')
    test_transaction = pd.read_csv(os.path.join(DATA_DIRECTORY, 'test_transaction.csv'), index_col='TransactionID')

    submission = pd.read_csv(os.path.join(DATA_DIRECTORY, 'sample_submission.csv'))

    return train_identity, train_transaction, rename_test_identity_columns(test_identity), test_transaction, submission
