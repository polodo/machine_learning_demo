from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy
from keras.models import model_from_json
from mysite.prediction.neural_network import *
import time


"""
This is a demo that I use LSTM to train model used for Bitcoin price prediction.
It will have 2 parts that I will do
1. Crawling data from Binance for closed price for each period (usually use 30mins).
    I will crawl data of 500 nearest days.
    Each record is closed price in 30mins.
    Total 24408 prices.
2. Use ML to build model that it fit with given data.
    - Step 1: Preprocess raw data above.
    - Step 2: Build Model
    - Step 3: Training Model and Evaluate Model, then save it for later use
    - Step 4: Plot predictions on graph to visualize result.
    - Step 5: Plot predictions trend in next 5 days to graph
 
"""

INTERVAL = 30 # int, minutes for time series
window_size = 50 # window size for each record set

BUILD_AND_PREDICT = False
model = Sequential()
f = open('BTCUSDT_30mins.csv', 'r').read()
data = [float(d) for d in f.split('\n')]
print(min(data[-1030:]), max(data[-1030:]), (min(data[-1030:]) + max(data[-1030:])) / 2)

# plot_graph_for_timeseries(data)

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


#future unknown predictions: in this case, test_set doesn't exist

future_pred_count = 5 * 24 * 2 #let's predict 100 new steps

model.reset_states() #always reset states when inputting a new sequence

#first, let set the model's states (it's important for it to know the previous trends)
new_points = predict_sequences_multiple(model, X_test, window_size, future_pred_count)

#after processing a sequence, reset the states for safety
model.reset_states()
future_predictions = numpy.append(predictions, new_points[-1])
# future_predictions = numpy.append([0] * len(predictions), new_points[-1])
plot_results_multiple(future_predictions, y_test, future_pred_count)