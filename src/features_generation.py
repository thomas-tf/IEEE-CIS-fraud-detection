import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
from utils import load_data
from recursive_feature_elimiation import recursive_feature_elimination

DATA_DIRECTORY = '../input'


def id_split(df):
    """
    Split or convert some identifiable features in subsets
    :param df:
    :return df:
    """
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

    df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]

    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]

    df['id_34'] = df['id_34'].str.split(':', expand=True)[1]
    df['id_23'] = df['id_23'].str.split(':', expand=True)[1]

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[
                                   df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    df['had_id'] = 1
    gc.collect()

    return df


def merge_transaction_and_identify(transaction, identity):
    df = transaction.merge(identity, how='left', left_index=True, right_index=True)

    del transaction, identity

    gc.collect()

    return df


def email_mappings(train, test):
    """
    group email domains to company
    yahoo / ymail / frontier / rocketmail -> Yahoo
    hotmail / outlook / live / msn -> Microsoft
    icloud / mac / me -> Appe
    prodigy / att / sbcglobal-> AT&T
    centurylink / embarqmail / q -> Centurylink
    aim / aol -> AOL
    twc / charter -> Spectrum
    :param train:
    :param test:
    :return train, test:
    """
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
              'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo',
              'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',
              'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google',
              'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
              'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft',
              'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
              'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other',
              'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft',
              'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other',
              'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo',
              'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
              'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

    us_emails = ['gmail', 'net', 'edu']

    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)

        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    return train, test


def transaction_amt_features(train, test):
    """
    generate features on base on transaction amount
    :param train:
    :param test:
    :return train, test:
    """
    # log transformation
    train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
    test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

    # get decimal part as feature
    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(
        int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

    return train, test


def drop_useless_columns(train, test):
    """
    drop columns if:
    - only 1 category
    - More than 90% of the values are NaN
    - More than 90% of the values are the same
    :param train:
    :param test:
    :return train, test:
    """
    one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

    many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
    many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

    big_top_value_cols = [col for col in train.columns if
                          train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    big_top_value_cols_test = [col for col in test.columns if
                               test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    cols_to_drop = list(set(
        many_null_cols +
        many_null_cols_test +
        big_top_value_cols +
        big_top_value_cols_test +
        one_value_cols +
        one_value_cols_test
    ))

    # exclude target
    cols_to_drop.remove('isFraud')

    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)

    return train, test


def top_features_aggregation(train, test):
    """
    aggregate top ranked features (by RFE)
    :param train:
    :param test:
    :return train, test:
    """
    columns_a = ['TransactionAmt', 'id_02', 'D15']
    columns_b = ['card1', 'card4', 'addr1']

    for col_a in columns_a:
        for col_b in columns_b:
            for df in [train, test]:
                df[f'{col_a}_to_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')
                df[f'{col_a}_to_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('std')

    return train, test


def label_encoding(train, test):
    """
    Label encode categorical columns
    :param train:
    :param test:
    :return:
    """
    for col in train.columns:
        if train[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))
            test[col] = le.transform(list(test[col].astype(str).values))

    return train, test


def clean_inf_nan(df):
    """
    replace infs to nan
    reference: https://www.kaggle.com/dimartinot
    :param df:
    :return df:
    """
    return df.replace([np.inf, -np.inf], np.nan)


def feature_engineering():
    """
    load original datasets and conduct feature engineering
    :return X_train, y_train, X_test, submission:
    """
    train_identity, train_transaction, test_identity, test_transaction, submission = load_data(DATA_DIRECTORY)

    train_identity = id_split(train_identity)
    test_identity = id_split(test_identity)

    train = merge_transaction_and_identify(train_transaction, train_identity)
    test = merge_transaction_and_identify(test_transaction, test_identity)

    train, test = email_mappings(train, test)

    train, test = drop_useless_columns(train, test)

    useful_features = recursive_feature_elimination(train)

    cols_to_drop = [col for col in train.columns if col not in useful_features]
    cols_to_drop.remove('isFraud')
    cols_to_drop.remove('TransactionDT')

    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)

    train, test = label_encoding(train, test)

    X_train = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
    y_train = train.sort_values('TransactionDT')['isFraud']

    X_test = test.drop(['TransactionDT'], axis=1)

    del train, test
    gc.collect()

    X_train = clean_inf_nan(X_train)
    X_test = clean_inf_nan(X_test)

    return X_train, y_train, X_test, submission
