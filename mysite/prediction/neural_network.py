import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROWS_FOR_PREDICT = 10 # int, base number to get record for predict and prove model, example num_rows = len(data) / ROWS_FOR_PREDICT
MAX_ROWS_FOR_PREDICT = 1000

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(211)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    # print(predicted_data)
    ax.plot(range(len(predicted_data[0: len(predicted_data) - prediction_len])), [float(d) for d in predicted_data[0: len(predicted_data) - prediction_len]])
    new_data = [None] * len(predicted_data[0: len(predicted_data) - prediction_len])
    new_x = [x for x in range(len(predicted_data[0: len(predicted_data) - prediction_len]))]
    for data in predicted_data[-prediction_len:]:
        new_data.append(data)
        new_x.append(len(new_x) + 1)
    ax.plot(new_x, new_data)
    # find lowest point
    min_p = [0, 0]
    max_p = [0, 0]
    for i, d in enumerate(predicted_data):
        if d < min_p[-1]:
            min_p = [i, d]
            continue
        if d > max_p[-1]:
            max_p = [i, d]
    ax.plot(min_p[0], min_p[1],"o")
    ax.text(min_p[0] + 0.025, min_p[1] - 0.025, "Price = %f" % (11106.215 + min_p[1] * 11106.215))
    ax.plot(max_p[0], max_p[1], "*")
    ax.text(max_p[0] + 0.025, max_p[1] - 0.025, "Price = %f" % (11106.215 + max_p[1] * 11106.215))
    # for i, data in enumerate(predicted_data):
    #     padding = [None for p in range(i * prediction_len)]
    #     plt.plot(padding + data, label='Prediction')
    #     plt.legend()
    plt.show()

def plot_graph_for_timeseries(data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(212)
    ax.plot(range(len(data)), [float(d) for d in data])
    plt.show()

def load_data(filename, seq_len, normalise_window, predict_demo=False):
    f = open(filename, 'r').read()
    data = f.split('\n')
    sequence_length = seq_len + 1
    result = []
    data_for_demo = []
    if not predict_demo:
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
    else:
        lenght_predict_demo = max(int(len(data) / ROWS_FOR_PREDICT), MAX_ROWS_FOR_PREDICT)
        for index in range(len(data) - sequence_length - lenght_predict_demo):
            result.append(data[index: index + sequence_length])
        for index in range(lenght_predict_demo - sequence_length):
            data_for_demo.append(data[len(data) - lenght_predict_demo + index:  index + sequence_length])
    if normalise_window:
        result = normalise_windows(result)
        if predict_demo:
            data_for_demo = np.array(normalise_windows(data_for_demo))

    result = np.array(result)

    row = round(0.9577 * result.shape[0])
    if predict_demo:
        print (data_for_demo.shape[0])
        print (len(data_for_demo), data_for_demo)
        data_for_demo = data_for_demo[:int(round(0.9577 * data_for_demo.shape[0])), :]
        np.random.shuffle(data_for_demo)
        data_for_demo = data_for_demo[:, :-1]
        data_for_demo = np.reshape(data_for_demo, (data_for_demo.shape[0], data_for_demo.shape[1], 1))
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print("*** CROSS VALIDATION *** ")
    print("Training input: %d records. " % (len(x_train)))
    print("Training output: %d records. " % (len(y_train)))
    print("Test input: %d records. " % (len(x_test)))
    print("Test output: %d records. " % (len(y_test)))
    print("*** END CROSS VALIDATION *** ")

    return [x_train, y_train, x_test, y_test, data_for_demo]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of X steps before shifting prediction run forward by X steps
    prediction_seqs = []
    for i in range(0, int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            pred_y = model.predict(curr_frame[newaxis, :, :])
            # print("curr_frame = ", curr_frame[newaxis, :, :])
            # print ("PRED.Y = ", pred_y)
            predicted.append(pred_y[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def create_model(stateful, lahead, batch_size):
    model = Sequential()
    model.add(LSTM(20,
              input_shape=(lahead, 1),
              batch_size=batch_size,
              stateful=stateful))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model