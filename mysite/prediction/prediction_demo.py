from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from mysite.prediction.neural_network import *
import time

INTERVAL = 30 # int, minutes for time series
window_size = 50 # window size for each record set

BUILD_AND_PREDICT = True
model = Sequential()
f = open('BTCUSDT_30mins.csv', 'r').read()
data = f.split('\n')
plot_graph_for_timeseries(data)

#Step 1 Load Data
X_train, y_train, X_test, y_test, data_for_demo = load_data('BTCUSDT_30mins.csv', window_size, True, False)

if not BUILD_AND_PREDICT:
    # load json and create model
    json_file = open('model_3days.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_3days.h5")
    print("Loaded model from disk")
else:
    #Step 2 Build Model
    model.add(LSTM(
        input_dim=1,
        output_dim=window_size,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        window_size,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='adam')
    print ('compilation time : ', time.time() - start)

    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=window_size,
        nb_epoch=1,
        validation_split=0.05)

    # evaluate the model by real data
    scores = model.evaluate(X_test, y_test, verbose=0)
    # Step 3.1 dump model to local file to reuse
    model_json = model.to_json()
    with open("model_3days.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_3days.h5")
    print("Saved model to disk")

# Step 4 - Plot the predictions!
# predictions = predict_sequences_multiple(model, X_test, window_size, len(y_test))
predictions = predict_point_by_point(model, X_test)
plot_results_multiple(predictions, y_test, len(y_test))
