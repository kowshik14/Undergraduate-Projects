import pandas as pd
from scipy.io import arff

def load_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    # Convert target to integer
    df['target'] = df['target'].apply(lambda x: int(x.decode('utf-8')))
    return df

def preprocess_data():
    hEOG_train = load_data('./data/EOGHorizontalSignal_TRAIN.arff')
    hEOG_test = load_data('./data/EOGHorizontalSignal_TEST.arff')
    vEOG_train = load_data('./data/EOGVerticalSignal_TRAIN.arff')
    vEOG_test = load_data('./data/EOGVerticalSignal_TEST.arff')

    y_train = hEOG_train['target'].copy()
    y_test = hEOG_test['target'].copy()

    #X_train_combined = pd.concat([hEOG_train.drop(columns=['target']), vEOG_train.drop(columns=['target'])], axis=1)
    #X_test_combined = pd.concat([hEOG_test.drop(columns=['target']), vEOG_test.drop(columns=['target'])], axis=1)

    return hEOG_train, hEOG_test, vEOG_train, vEOG_test, y_train, y_test

def export_features(combined_features , y, filename):
    df_features= pd.DataFrame(combined_features)
    df_features['target'] = y

    return df_features.to_csv(f'./results/{filename}.csv', index=False)