def converttoWinPath(s):
    s=s.replace('\\','\\\\')
    return s

DATASET_PATH=""
dl=input("Have you downloaded the necessary dataset?(Y/N)")
if dl=='N' or dl=='n':
    import extract
    extract
    DATASET_PATH=extract.folder_selected
else:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    from tkinter import *
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("Select the folder where dataset is located")
    root = Tk()
    root.withdraw()
    DATASET_PATH = filedialog.askdirectory()
    TRAIN_DIR=DATASET_PATH+"\Training"
    TRAIN_DIR=converttoWinPath(TRAIN_DIR)           #cause I cant use \ as its an escape sequence in VS
    VAL_DIR=DATASET_PATH+"\Test"
    VAL_DIR=converttoWinPath(VAL_DIR)

    train_datagen= ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,    #image augmentation for training dataset
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    val_datagen= ImageDataGenerator( rescale = 1./255. ) #we dont need to augment validation dataset
    print("Processing datasets...")
    train_gen= train_datagen.flow_from_directory(TRAIN_DIR,
                                                 batch_size = 20,
                                                 class_mode = 'categorical', 
                                                 target_size = (100, 100)) 
    val_gen= val_datagen.flow_from_directory(VAL_DIR,
                                              batch_size  = 20,
                                              class_mode  = 'categorical', 
                                              target_size = (100, 100))
    print("Compiling Model")
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(114, activation=tf.nn.softmax)])

    model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

    model.summary()

    history = model.fit_generator(
            train_gen,
            validation_data = val_gen,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 10,
            verbose = 1)
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

