from tensorflow_core.python.keras import Sequential, regularizers
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_core.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


class TaggerModel:
    def __init__(self):
        self.feature_dim_1 = 1
        self.feature_dim_2 = 1
        self.channel = 1
        self.num_classes = 12

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu',
                         input_shape=(self.feature_dim_1, self.feature_dim_2, self.channel)))
        model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
        model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPool2D(pool_size=(1, 1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
        model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPool2D(pool_size=(1, 1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
        model.add(Conv2D(256, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256, kernel_regularizer=regularizers.l2(0.2), activation='relu'))
        model.add(Dense(32, kernel_regularizer=regularizers.l2(0.2), activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
        return model

    @staticmethod
    def get_callbacks():
        callbacks = [
            ModelCheckpoint(filepath='tagger.model.best.hdf5',
                            verbose=1, save_best_only=True),
            EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor='val_loss',
                # "no longer improving" being defined as "no better than 1e-3 less"
                min_delta=1e-3,
                # "no longer improving" being further defined as "for at least 4 epochs"
                patience=4,
                verbose=1),
        ]
        return callbacks
