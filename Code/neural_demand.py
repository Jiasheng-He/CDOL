import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import ShuffleSplit
from keras.callbacks import EarlyStopping

# Generate dummy data
def nerual_demand():
    df_list = [pd.read_table('../data/basic_tiny_2.csv', sep=' ', index_col=None),]
               #pd.read_table('../data/processed/final_taxi_day2.csv', sep=' ', index_col=None)]
    data_df = pd.concat(df_list, axis=0, ignore_index=True)
    data_df.drop_duplicates(['Hour', 'Min', 'Gid'], inplace=True)
    data_df = data_df.reset_index(drop=True)


    feature_list = list(data_df.columns.values)
    feature_list.remove('Car1r')

    ssplit = ShuffleSplit(n_splits=1, test_size=.8, random_state=0)
    for train_index, test_index in ssplit.split(data_df):
        train_df = data_df.iloc[train_index]
        test_df = data_df.iloc[test_index]

    x_train = train_df[feature_list]
    y_train = train_df['Car1r']

    x_test = test_df[feature_list]
    y_test = test_df['Car1r']

    model = Sequential()
    model.add(Dense(64, input_dim=len(feature_list), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,validation_split=0.2, callbacks=[early_stopping],
              epochs=1,
              batch_size=128)

    score = model.predict(x_test, batch_size=25)
    fabs_sum = 0
    label_sum = 0
    y_test = list(y_test)
    for i in range(len(y_test)):
        fabs_sum += np.fabs(score[i][0] - y_test[i])
        label_sum += y_test[i]

    print(fabs_sum/label_sum)