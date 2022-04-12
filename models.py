import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.layers import *
from xlwt import Workbook
from data_preprocessing import duration


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


def run_lstm(lr, bs, ephs, scaler, x_train, y_train, x_test, y_test, working_dir):
    ## Creating the prelimaries
    model_name = 'LSTM'
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
    simple_lstm.add(keras.layers.Dense(duration))

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

    print('-----starting attention model-----')
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
    atten_lstm.add(keras.layers.Dense(duration))

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


def run_cnnlstm(lr, bs, ephs, scaler, x_train, y_train, x_test, y_test, working_dir):
    ## Creating the prelimaries
    model_name = 'CNNLSTM'
    filepath_simple = working_dir / Path(f'simple_{model_name}.hdf5')
    filepath_attention = working_dir / Path(f'attention_{model_name}.hdf5')

    checkpoint_simple = keras.callbacks.ModelCheckpoint(filepath_simple, monitor='val_loss', save_best_only=True)
    checkpoint_attention = keras.callbacks.ModelCheckpoint(filepath_attention, monitor='val_loss', save_best_only=True)

    wk=Workbook()
    sheet1 = wk.add_sheet('Simple', cell_overwrite_ok=True)
    sheet2 = wk.add_sheet('Attention', cell_overwrite_ok=True)
    sheet3 = wk.add_sheet('Real-test', cell_overwrite_ok=True)
    sheet4 = wk.add_sheet('Predicted-test', cell_overwrite_ok=True)

    ## Simple CNN-LSTM model
    K.clear_session()
    simple_cnnlstm = keras.Sequential()
    simple_cnnlstm.add(keras.layers.Conv1D(64, kernel_size=3, input_shape=(x_train.shape[1],x_train.shape[2])))
    simple_cnnlstm.add(keras.layers.Conv1D(64, kernel_size=3))
    simple_cnnlstm.add(keras.layers.LSTM(64, return_sequences=True))
    simple_cnnlstm.add(keras.layers.LSTM(64, return_sequences=True))
    simple_cnnlstm.add(keras.layers.Flatten())
    simple_cnnlstm.add(keras.layers.Dense(512, activation='relu'))
    simple_cnnlstm.add(keras.layers.Dense(128, activation='relu'))
    simple_cnnlstm.add(keras.layers.Dense(64, activation='relu'))
    simple_cnnlstm.add(keras.layers.Dense(32))
    simple_cnnlstm.add(keras.layers.Dense(duration))

    simple_cnnlstm.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = simple_cnnlstm.fit(x_train, y_train, validation_split=0.1, batch_size=bs, epochs=ephs, callbacks=[checkpoint_simple])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'simple_{model_name}.png'))
    plt.close()

    simple_cnnlstm.load_weights(filepath_simple)
    preds = simple_cnnlstm.predict(x_test)

    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(preds)

    for i in range(y_test.shape[1]):
        sheet1.write(0, 0, 'MSE')
        sheet1.write(0, 1, 'Hours Ahead')
        sheet1.write(i + 1, 0, mean_squared_error(y_test_unscaled[:,i],y_pred_unscaled[:,i]))
        sheet1.write(i + 1, 1, i+1)

    print('-----starting attention model-----')

    ## Attention model
    K.clear_session()
    atten_cnnlstm = keras.Sequential()
    atten_cnnlstm.add(keras.layers.Conv1D(64, kernel_size=3, input_shape=(x_train.shape[1],x_train.shape[2])))
    atten_cnnlstm.add(keras.layers.Conv1D(64, kernel_size=3))
    atten_cnnlstm.add(keras.layers.LSTM(64, return_sequences=True))
    atten_cnnlstm.add(keras.layers.LSTM(64, return_sequences=True))
    atten_cnnlstm.add(attention(return_sequences=True))
    atten_cnnlstm.add(keras.layers.Flatten())
    atten_cnnlstm.add(keras.layers.Dense(512, activation='relu'))
    atten_cnnlstm.add(keras.layers.Dense(128, activation='relu'))
    atten_cnnlstm.add(keras.layers.Dense(64, activation='relu'))
    atten_cnnlstm.add(keras.layers.Dense(32))
    atten_cnnlstm.add(keras.layers.Dense(duration))

    atten_cnnlstm.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    history = atten_cnnlstm.fit(x_train,y_train,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_attention])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'attention_{model_name}.png'))
    plt.close()

    atten_cnnlstm.load_weights(filepath_attention)
    preds = atten_cnnlstm.predict(x_test)

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


def run_convlstm(lr, bs, ephs, scaler, x_train, y_train, x_test, y_test, working_dir):
    ## Creating the prelimaries
    model_name = 'CONVLSTM'
    filepath_simple = working_dir / Path(f'simple_{model_name}.hdf5')
    filepath_attention = working_dir / Path(f'attention_{model_name}.hdf5')

    checkpoint_simple = keras.callbacks.ModelCheckpoint(filepath_simple, monitor='val_loss', save_best_only=True)
    checkpoint_attention = keras.callbacks.ModelCheckpoint(filepath_attention, monitor='val_loss', save_best_only=True)

    wk=Workbook()
    sheet1 = wk.add_sheet('Simple', cell_overwrite_ok=True)
    sheet2 = wk.add_sheet('Attention', cell_overwrite_ok=True)
    sheet3 = wk.add_sheet('Real-test', cell_overwrite_ok=True)
    sheet4 = wk.add_sheet('Predicted-test', cell_overwrite_ok=True)

    ## Reshaping the training and testing data to suit the convlstm model
    x_train_conv =x_train.reshape(x_train.shape[0], 1, 1, x_train.shape[1], x_train.shape[2])
    x_test_conv = x_test.reshape(x_test.shape[0], 1, 1, x_test.shape[1], x_test.shape[2])

    ## Simple ConvLSTM model
    K.clear_session()
    simple_convlstm = keras.Sequential()
    simple_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True, 
                                                input_shape=(x_train_conv.shape[1], x_train_conv.shape[2], 
                                                            x_train_conv.shape[3], x_train_conv.shape[4])))
    simple_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True))
    simple_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True))
    simple_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True))
    simple_convlstm.add(keras.layers.Flatten())
    simple_convlstm.add(keras.layers.Dense(512, activation='relu'))
    simple_convlstm.add(keras.layers.Dense(128, activation='relu'))
    simple_convlstm.add(keras.layers.Dense(64, activation='relu'))
    simple_convlstm.add(keras.layers.Dense(32))
    simple_convlstm.add(keras.layers.Dense(duration))

    simple_convlstm.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = simple_convlstm.fit(x_train_conv,y_train,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_simple])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'simple_{model_name}.png'))
    plt.close()

    simple_convlstm.load_weights(filepath_simple)
    preds = simple_convlstm.predict(x_test_conv)

    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(preds)

    for i in range(y_test.shape[1]):
        sheet1.write(0, 0, 'MSE')
        sheet1.write(0, 1, 'Hours Ahead')
        sheet1.write(i + 1, 0, mean_squared_error(y_test_unscaled[:,i],y_pred_unscaled[:,i]))
        sheet1.write(i + 1, 1, i+1)

    print('-----starting attention model-----')

    ## Attention model
    K.clear_session()
    atten_convlstm = keras.Sequential()
    atten_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True, 
                                                input_shape=(x_train_conv.shape[1], x_train_conv.shape[2], 
                                                            x_train_conv.shape[3], x_train_conv.shape[4])))
    atten_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True))
    atten_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True))
    atten_convlstm.add(keras.layers.ConvLSTM2D(64, kernel_size=(1,2),return_sequences=True))
    atten_convlstm.add(attention(return_sequences=True))
    atten_convlstm.add(keras.layers.Flatten())
    atten_convlstm.add(keras.layers.Dense(512, activation='relu'))
    atten_convlstm.add(keras.layers.Dense(128, activation='relu'))
    atten_convlstm.add(keras.layers.Dense(64, activation='relu'))
    atten_convlstm.add(keras.layers.Dense(32))
    atten_convlstm.add(keras.layers.Dense(duration))

    atten_convlstm.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = atten_convlstm.fit(x_train_conv,y_train,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_attention])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'attention_{model_name}.png'))
    plt.close()

    atten_convlstm.load_weights(filepath_attention)
    preds = atten_convlstm.predict(x_test_conv)

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


def run_seq2seq(lr, bs, ephs, scaler, x_train, y_train, x_test, y_test, working_dir):
    ## Creating the prelimaries
    model_name = 'SEQ2SEQ'
    filepath_simple = working_dir / Path(f'simple_{model_name}.hdf5')
    filepath_attention = working_dir / Path(f'attention_{model_name}.hdf5')

    checkpoint_simple = keras.callbacks.ModelCheckpoint(filepath_simple, monitor='val_loss', save_best_only=True)
    checkpoint_attention = keras.callbacks.ModelCheckpoint(filepath_attention, monitor='val_loss', save_best_only=True)

    wk=Workbook()
    sheet1 = wk.add_sheet('Simple', cell_overwrite_ok=True)
    sheet2 = wk.add_sheet('Attention', cell_overwrite_ok=True)
    sheet3 = wk.add_sheet('Real-test', cell_overwrite_ok=True)
    sheet4 = wk.add_sheet('Predicted-test', cell_overwrite_ok=True)

    ## Reshaping the training data to suit Seq2Seq model
    y_train_seq = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

    ## Simple Model
    K.clear_session()
    input_train = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    output_train = keras.layers.Input(shape=(y_train_seq.shape[1], y_train_seq.shape[2]))

    ### --------------------------------Encoder Section -----------------------------------------------###
    encoder_first = keras.layers.LSTM(128, return_sequences=True, return_state=False)(input_train)
    encoder_second = keras.layers.LSTM(128, return_sequences=True)(encoder_first)
    encoder_third = keras.layers.LSTM(128, return_sequences=True)(encoder_second)
    encoder_fourth, encoder_fourth_s1, encoder_fourth_s2 = keras.layers.LSTM(128,return_sequences=False, return_state=True)(encoder_third)

    ###---------------------------------Decorder Section-----------------------------------------------###
    decoder_first = keras.layers.RepeatVector(output_train.shape[1])(encoder_fourth)
    decoder_second = keras.layers.LSTM(128, return_state=False, return_sequences=True)(decoder_first,initial_state=[encoder_fourth,encoder_fourth_s2])
    decoder_third = keras.layers.LSTM(128,return_sequences=True)(decoder_second)
    decoder_fourth = keras.layers.LSTM(128,return_sequences=True)(decoder_third)
    decoder_fifth = keras.layers.LSTM(128,return_sequences=True)(decoder_fourth)
    # print(decoder_fifth)

    ###--------------------------------Output Section-------------------------------------------------###
    output = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_fifth)

    simple_seq = keras.Model(inputs=input_train, outputs=output)
    opt = keras.optimizers.Adam(learning_rate=lr)
    simple_seq.compile(loss='mse', optimizer=opt, metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = simple_seq.fit(x_train,y_train_seq,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_simple])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'simple_{model_name}.png'))
    plt.close()

    simple_seq.load_weights(filepath_simple)
    preds = simple_seq.predict(x_test)

    preds = preds.reshape(preds.shape[0],preds.shape[1])

    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(preds)

    for i in range(y_test.shape[1]):
        sheet1.write(0, 0, 'MSE')
        sheet1.write(0, 1, 'Hours Ahead')
        sheet1.write(i + 1, 0, mean_squared_error(y_test_unscaled[:,i],y_pred_unscaled[:,i]))
        sheet1.write(i + 1, 1, i+1)

    print('-----starting attention model-----')

    ## Attention Model
    K.clear_session()

    input_train = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    output_train = keras.layers.Input(shape=(y_train_seq.shape[1], y_train_seq.shape[2]))

    ###----------------------------------------Encoder Section------------------------------------------###
    encoder_first = keras.layers.LSTM(128, return_sequences=True, return_state=False)(input_train)
    encoder_second = keras.layers.LSTM(128, return_sequences=True)(encoder_first)
    encoder_third = keras.layers.LSTM(128, return_sequences=True)(encoder_second)
    encoder_fourth, encoder_fourth_s1, encoder_fourth_s2 = keras.layers.LSTM(128,return_sequences=True,return_state=True)(encoder_third)

    ###-----------------------------------------Decoder Section------------------------------------------###
    decoder_first = keras.layers.RepeatVector(output_train.shape[1])(encoder_fourth_s1)
    decoder_second = keras.layers.LSTM(128, return_state=False, return_sequences=True)(decoder_first, initial_state=[encoder_fourth_s1, encoder_fourth_s2])

    attention = keras.layers.dot([decoder_second, encoder_fourth], axes=[2, 2])
    attention = keras.layers.Activation('softmax')(attention)
    context = keras.layers.dot([attention, encoder_fourth], axes=[2, 1])

    decoder_third = keras.layers.concatenate([context, decoder_second])

    decoder_fourth = keras.layers.LSTM(128, return_sequences=True)(decoder_third)
    decoder_fifth = keras.layers.LSTM(128, return_sequences=True)(decoder_fourth)
    decoder_sixth = keras.layers.LSTM(128, return_sequences=True)(decoder_fifth)

    ###-----------------------------------------Output Section-----------------------------------------###
    output = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_sixth)

    atten_seq = keras.Model(inputs=input_train, outputs=output)
    opt = keras.optimizers.Adam(learning_rate=lr)
    atten_seq.compile(loss='mse', optimizer=opt, metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = atten_seq.fit(x_train,y_train_seq,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_attention])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'attention_{model_name}.png'))
    plt.close()

    atten_seq.load_weights(filepath_attention)
    preds = atten_seq.predict(x_test)

    preds = preds.reshape(preds.shape[0],preds.shape[1])

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


def run_wavenet(lr, bs, ephs, scaler, x_train, y_train, x_test, y_test, working_dir):
    ## Creating the prelimaries
    model_name = 'WAVENET'
    filepath_simple = working_dir / Path(f'simple_{model_name}.hdf5')
    filepath_attention = working_dir / Path(f'attention_{model_name}.hdf5')

    checkpoint_simple = keras.callbacks.ModelCheckpoint(filepath_simple, monitor='val_loss', save_best_only=True)
    checkpoint_attention = keras.callbacks.ModelCheckpoint(filepath_attention, monitor='val_loss', save_best_only=True)

    wk=Workbook()
    sheet1 = wk.add_sheet('Simple', cell_overwrite_ok=True)
    sheet2 = wk.add_sheet('Attention', cell_overwrite_ok=True)
    sheet3 = wk.add_sheet('Real-test', cell_overwrite_ok=True)
    sheet4 = wk.add_sheet('Predicted-test', cell_overwrite_ok=True)

    ## Simple Model
    K.clear_session()
    n_filters = 128
    filter_width = 2
    dilation_rates = [2**i for i in range(7)]

    inputs = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2]))
    x=inputs

    skips = []
    for dilation_rate in dilation_rates:

        x   = keras.layers.Conv1D(64, 1, padding='same')(x) 
        x_f = keras.layers.Conv1D(filters=n_filters,kernel_size=filter_width,padding='causal',dilation_rate=dilation_rate)(x)
        x_g = keras.layers.Conv1D(filters=n_filters,kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)

        z = keras.layers.Multiply()([keras.layers.Activation('tanh')(x_f),keras.layers.Activation('sigmoid')(x_g)])

        z = keras.layers.Conv1D(64, 1, padding='same', activation='relu')(z)

        x = keras.layers.Add()([x, z])    

        skips.append(z)

    out = keras.layers.Activation('relu')(keras.layers.Add()(skips)) 
    out = keras.layers.Conv1D(128, 1, padding='same')(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Dropout(0.4)(out)
    out = keras.layers.Conv1D(1, 1, padding='same')(out)

    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(duration)(out)

    simple_wavenet = keras.Model(inputs, out)

    simple_wavenet.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])


    ## Saving the result file to the folder of the model
    history = simple_wavenet.fit(x_train,y_train,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_simple])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'simple_{model_name}.png'))
    plt.close()

    simple_wavenet.load_weights(filepath_simple)
    preds = simple_wavenet.predict(x_test)

    preds = preds.reshape(preds.shape[0],preds.shape[1])
    
    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(preds)

    for i in range(y_test.shape[1]):
        sheet1.write(0, 0, 'MSE')
        sheet1.write(0, 1, 'Hours Ahead')
        sheet1.write(i + 1, 0, mean_squared_error(y_test_unscaled[:,i],y_pred_unscaled[:,i]))
        sheet1.write(i + 1, 1, i+1)

    print('-----starting attention model-----')

    ## Attention model
    K.clear_session()
    n_filters = 128
    filter_width = 2
    dilation_rates = [2**i for i in range(7)]

    inputs = Input(shape=(x_train.shape[1],x_train.shape[2]))
    x=inputs

    skips = []
    for dilation_rate in dilation_rates:

        x   = Conv1D(64, 1, padding='same')(x) 
        x_f = Conv1D(filters=n_filters,kernel_size=filter_width,padding='causal',dilation_rate=dilation_rate)(x)
        x_g = Conv1D(filters=n_filters,kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)

        z = Multiply()([keras.layers.Activation('tanh')(x_f),keras.layers.Activation('sigmoid')(x_g)])

        z = Conv1D(64, 1, padding='same', activation='relu')(z)

        x = Add()([x, z])    

        skips.append(z)

    out = Activation('relu')(keras.layers.Add()(skips)) 
    #out = attention(return_sequences=True)(out)
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(0.4)(out)
    out = Conv1D(1, 1, padding='same')(out)
    #out = attention(return_sequences=True)(out)
    out = Flatten()(out)
    out = Dense(duration)(out)

    atten_wavenet = keras.Model(inputs, out)

    atten_wavenet.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

    ## Saving the result file to the folder of the model
    history = atten_wavenet.fit(x_train,y_train,validation_split=0.1,batch_size=bs,epochs=ephs,callbacks=[checkpoint_attention])

    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.legend()
    plt.savefig(working_dir / Path(f'attention_{model_name}.png'))
    plt.close()

    atten_wavenet.load_weights(filepath_attention)
    preds = atten_wavenet.predict(x_test)

    preds = preds.reshape(preds.shape[0],preds.shape[1])
    
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
