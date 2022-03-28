import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from xlwt import Workbook


class attention(keras.layers.Layer):
    '''
    Attention layer for the neural networks.
    
    if return_sequences=True, it will give 3D vector and if false it will give 2D vector. It is same as LSTMs.

    https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    the  following code is being inspired from the above link.
    '''

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)


def run_lstm(model_name, lr, bs, ephs, scaler, x_train, y_train, x_test, y_test, working_dir):
    ## Creating the prelimaries
    filepath_simple = working_dir / Path(f'simple_{model_name}.hdf5')
    filepath_attention = working_dir / Path(f'attention_{model_name}.hdf5')

    checkpoint_simple = keras.callbacks.ModelCheckpoint(filepath_simple, monitor='val_loss', save_best_only=True)
    checkpoint_attention = keras.callbacks.ModelCheckpoint(filepath_attention, monitor='val_loss', save_best_only=True)

    wk=Workbook()
    sheet1 = wk.add_sheet('Simple', cell_overwrite_ok=True)
    sheet2 = wk.add_sheet('Attention', cell_overwrite_ok=True)
    sheet3 = wk.add_sheet('Real-test', cell_overwrite_ok=True)
    sheet4 = wk.add_sheet('Predicted-test', cell_overwrite_ok=True)

    ## Simple LSTM
    K.clear_session()
    simple_lstm = keras.Sequential()
    simple_lstm.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    simple_lstm.add(keras.layers.LSTM(64, return_sequences=True))
    simple_lstm.add(keras.layers.Dropout(0.3))
    simple_lstm.add(keras.layers.LSTM(64, return_sequences=True))
    simple_lstm.add(keras.layers.LSTM(64, return_sequences=True))
    simple_lstm.add(keras.layers.Flatten())
    simple_lstm.add(keras.layers.Dense(512, activation='relu'))
    simple_lstm.add(keras.layers.Dense(128, activation='relu'))
    simple_lstm.add(keras.layers.Dense(64, activation='relu'))
    simple_lstm.add(keras.layers.Dropout(0.3))
    simple_lstm.add(keras.layers.Dense(32))
    simple_lstm.add(keras.layers.Dense(6))

    simple_lstm.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = simple_lstm.fit(x_train, y_train, validation_split=0.1, batch_size=bs, epochs=ephs, callbacks=[checkpoint_simple])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'simple_{model_name}.png'))
    plt.close()

    simple_lstm.load_weights(filepath_simple)
    preds = simple_lstm.predict(x_test)

    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(preds)

    for i in range(y_test.shape[1]):
        sheet1.write(0, 0, 'MSE')
        sheet1.write(0, 1, 'Hours Ahead')
        sheet1.write(i + 1, 0, mean_squared_error(y_test_unscaled[:,i],y_pred_unscaled[:,i]))
        sheet1.write(i + 1, 1, i+1)

    ## Attention model
    K.clear_session()
    atten_lstm = keras.Sequential()
    atten_lstm.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    atten_lstm.add(keras.layers.LSTM(64, return_sequences=True))
    atten_lstm.add(keras.layers.Dropout(0.3))
    atten_lstm.add(keras.layers.LSTM(64, return_sequences=True))
    atten_lstm.add(keras.layers.LSTM(64, return_sequences=True))
    atten_lstm.add(attention(return_sequences=True))
    atten_lstm.add(keras.layers.Flatten())
    atten_lstm.add(keras.layers.Dense(512, activation='relu'))
    atten_lstm.add(keras.layers.Dense(128, activation='relu'))
    atten_lstm.add(keras.layers.Dense(64, activation='relu'))
    atten_lstm.add(keras.layers.Dropout(0.3))
    atten_lstm.add(keras.layers.Dense(32))
    atten_lstm.add(keras.layers.Dense(6))

    atten_lstm.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    history = atten_lstm.fit(x_train,y_train,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_attention])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'attention_{model_name}.png'))
    plt.close()

    atten_lstm.load_weights(filepath_attention)
    preds = atten_lstm.predict(x_test)

    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(preds)

    for i in range(y_test.shape[1]):
            sheet2.write(0, 0, 'MSE')
            sheet2.write(0, 1, 'Hours Ahead')
            sheet2.write(i + 1, 0, mean_squared_error(y_test_unscaled[:,i],y_pred_unscaled[:,i]))
            sheet2.write(i + 1, 1, i+1)
            sheet3.write(0, i, "Real-test")
            sheet4.write(0, i, "Predicted_test")
            for j in range(y_test_unscaled.shape[0]):
                    sheet3.write(j + 1, i, float(y_test_unscaled[j, i]))
            for k in range(y_pred_unscaled.shape[0]):
                    sheet4.write(k + 1, i, float(y_pred_unscaled[k, i]))   

    wk.save(working_dir / Path(f'{model_name} Result.xls'))
