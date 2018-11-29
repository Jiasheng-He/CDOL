import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics


def nerual():
    data_df = pd.read_table('../data/basic_tiny_2.csv', sep=' ', index_col=None).sample(frac=1).reset_index(drop=True)

    ssplit = ShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    for train_index, test_index in ssplit.split(data_df):
        train_df = data_df.iloc[train_index]
        test_df = data_df.iloc[test_index]


    feature_list = list(train_df.columns.values)
    feature_list.remove('Label')

    x_train = train_df[feature_list]
    y_train = train_df['Label'].map(lambda x: int(x))

    x_test = test_df[feature_list]
    y_test = test_df['Label'].map(lambda x: int(x))

    model = Sequential()
    model.add(Dense(64, input_dim=len(feature_list), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=1,
              batch_size=50)

    score = model.evaluate(x_test, y_test, batch_size=2500)
    score2 = model.predict(x_test, batch_size=2500)
    auc_score = metrics.roc_auc_score(y_test, score2)
    print([score, auc_score])