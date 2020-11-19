import os
import pandas as pd


def load_data(DATA_DIRECTORY):
    train_identity = pd.read_csv(os.path.join(DATA_DIRECTORY, 'train_identity.csv'), index_col='TransactionID')
    train_transaction = pd.read_csv(os.path.join(DATA_DIRECTORY, 'train_transaction.csv'), index_col='TransactionID')

    test_identity = pd.read_csv(os.path.join(DATA_DIRECTORY, 'test_identity.csv'), index_col='TransactionID')
    test_transaction = pd.read_csv(os.path.join(DATA_DIRECTORY, 'test_transaction.csv'), index_col='TransactionID')

    submission = pd.read_csv(os.path.join(DATA_DIRECTORY, 'sample_submission.csv'))

    return train_identity, train_transaction, test_identity, test_transaction, submission
