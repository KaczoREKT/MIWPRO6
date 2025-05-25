import pandas as pd
import keras as ks
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))

def show_dataset_info(pokedex):
    print(pokedex.info())
    print(pokedex.head())

def load_pokedex(description_file, image_folder):
    pokedex = pd.read_csv(description_file)
    pokedex.drop('Type2', axis=1, inplace=True)
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)
    images = sorted(os.listdir(image_folder))
    images = list(map(lambda image_file: os.path.join(image_folder, image_file), images))
    pokedex['Image']= images
    return pokedex

def prepare_dataset(pokedex):
    data_generator = ImageDataGenerator(validation_split=0.1, rescale=1.0/255,
                                        rotation_range=30, width_shift_range=0.5,
                                        height_shift_range=0.5, zoom_range=0.5,
                                        fill_mode='nearest')
    train_generator = data_generator.flow_from_dataframe(pokedex,
                        x_col='Image', y_col='Type1', subset='training',
                        color_mode='rgba', class_mode='categorical',
                        target_size=(120,120), shuffle=True, batch_size=32)
    test_generator = data_generator.flow_from_dataframe(pokedex,
                        x_col='Image', y_col='Type1', subset='validation',
                        color_mode='rgba', class_mode='categorical',
                        target_size=(120,120), shuffle=True)
    return train_generator, test_generator



def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--description_file', default='pokemon_dataset/pokemon.csv',
            help='csvfile with pokemon information')
    parser.add_argument('-i', '--image_folder', default='pokemon_dataset/images/images')
    parser.add_argument('-o', '--outfile', default='model.txt')
    parser.add_argument('-hist', '--history_image')
    parser.add_argument('-m', '--model_folder', default='model')
    parser.add_argument('-c', '--confusion_matrix', default='confusion_matrix.png')
    return parser.parse_args()

def prepare_network():
    model = ks.models.Sequential()
    model.add(ks.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120,120,4)))
    model.add(ks.layers.MaxPooling2D((2, 2)))
    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(ks.layers.MaxPooling2D((2, 2)))
    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(64, activation='relu'))
    model.add(ks.layers.Dense(18, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_accuracy(history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Dokładność modelu')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def evaluate_model(model, test_generator):
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    class_names = list(test_generator.class_indices.keys())
    display_labels = class_names[:len(np.unique(y_true))]

    print(classification_report(y_true, y_pred_classes, target_names=display_labels))

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.tight_layout()
    plt.show()

def main():
    args = parse_arguments()
    pokedex = load_pokedex(args.description_file, args.image_folder)
    show_dataset_info(pokedex)
    train_generator, test_generator = prepare_dataset(pokedex)
    model = prepare_network()
    history = model.fit(train_generator, validation_data=test_generator, epochs=30)
    model.save(args.model_folder)

    # Rysowanie wykresu dokładności
    plot_accuracy(history, save_path=args.history_image)

    # Ewaluacja modelu i macierz pomyłek
    evaluate_model(model, test_generator)

if __name__=='__main__':
    main()
