import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from src.models.utils import load_data, plot_confusion_matrix, evaluate_results, display_errors,load_data_with_diffent_ratio,load_data_with_diffent_P_N_Ratio

np.random.seed(0)
random_seed = 0
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sns.set(style='white', context='notebook', palette='deep')
different_sets = ['SC1', 'SC5', 'SC10']
# different_sets = ['TC1', 'TC5', 'TC10']

for dataset in different_sets:
    BATCH_SIZE = 8
    EPOCHS = 40
    KERNEL_SIZE = (6, 6)
    # MODEL_PATH = "../trained_models/cnn_trained_model.h5"
    # 1. Data preparation
    # Load data
    X_train, X_val, X_test, Y_train, Y_val,Y_test =load_data_with_diffent_P_N_Ratio('../../../data/processed_data/hour_trail/all.npz', dataset)
    # #Split training and validation set
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = test_size[1], random_state=random_seed)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Testing {}'.format(dataset))
    if dataset[0]=='T':
        MODEL_PATH="../trained_models/{}_cnn_without_augment_Test={}Val={}Train={}_EPOCHS={}_BATCH_SIZE={}_KERNEL={}.h5".format('SC1', 600, 600, 600, EPOCHS, BATCH_SIZE, KERNEL_SIZE)
    else:
        MODEL_PATH = "../trained_models/{}_cnn_without_augment_Test={}Val={}Train={}_EPOCHS={}_BATCH_SIZE={}_KERNEL={}.h5".format(dataset, Y_test.shape[0], Y_val.shape[0], Y_train.shape[0], EPOCHS, BATCH_SIZE, KERNEL_SIZE)
    # 2. CNN
    # set the model
    def CNN_model(batch_size = BATCH_SIZE,epochs = EPOCHS,conv_kernel_size = KERNEL_SIZE):
        model = Sequential()
        model.add(Conv2D(64, conv_kernel_size, padding='same', input_shape=(24, 244, 1), activation='relu'))
        model.add(Conv2D(128, conv_kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(64, conv_kernel_size, padding='same', input_shape=(24, 244, 1), activation='relu'))
        model.add(Conv2D(128, conv_kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        return model

    if not os.path.exists(MODEL_PATH):
        model = CNN_model()
        # set the optimizer and annealer
        optimizer = Adam(lr=0.0001, decay=1e-6)
        # complie the model
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                    patience=2,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        # Generators
        # # Without augmentation
        datagen = ImageDataGenerator()
        # With augmentation
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        #     zoom_range=0.2,  # Randomly zoom image
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=False,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images

        # Fit the model
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                      epochs=EPOCHS,
                                      validation_data=(X_val, Y_val),
                                      verbose=1,
                                      steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                      callbacks=[learning_rate_reduction])
        # 4. Evaluate the model
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
        ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)
        plt.show()
        # save model
        model.save(MODEL_PATH)
    else:
        print('Loading existed model from {}'.format(MODEL_PATH))
        model = keras.models.load_model(MODEL_PATH)



    # Predict the values from the validation dataset
    Y_val_pred = model.predict(X_val)
    # Convert predictions classes to one hot vectors
    Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_val_true = Y_val
    # compute the confusion matrix
    confusion_val_mtx = confusion_matrix(Y_val_true, Y_val_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_val_mtx, classes=range(2))
    print('------------Validation results evaluation------------')
    evaluate_results(Y_val_true, Y_val_pred)
    # display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
    display_errors(Y_val_pred_classes, Y_val_true,Y_val_pred,X_val)

    # predict results
    Y_test_pred = model.predict(X_test)
    Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)
    Y_test_true = Y_test
    print('------------Test results evaluation------------')
    evaluate_results(Y_test_true, Y_test_pred)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
