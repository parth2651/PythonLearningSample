#this will be Keras and TF version of TextColorPredictor

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import *

RUN_NAME = "run 1 with 3_3_4 node"

def main():

    training_data_df = pd.read_csv("Training_scaled.csv")
   
    X = training_data_df.drop(['y','yI'], axis=1).values
    Y = training_data_df[['y']].values

    
    # Define the model
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation='relu', name='layer_1'))
    model.add(Dense(3, activation='relu', name='layer_2'))
    #model.add(Dense(3, activation='relu', name='layer_3'))
    model.add(Dense(1, activation='linear', name='output_layer'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Create a TensorBoard logger
    logger = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(RUN_NAME),
        histogram_freq=0,
        write_graph=True
    )

    # Train the model
    model.fit(
        X,
        Y,
        epochs=100,
        shuffle=True,
        verbose=2,
        callbacks=[logger]
    )

    # Load the separate test data set
    test_data_df = pd.read_csv("Testing_scaled.csv")

    X_test = test_data_df.drop(['y','yI'], axis=1).values
    Y_test = test_data_df[['y']].values

    test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
    print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))



if __name__ == '__main__':
    main()