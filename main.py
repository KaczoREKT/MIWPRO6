import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    parser.add_argument('-c', '--confusion_matrix', default='confustion_matrix.png')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    pokedex = load_pokedex(args.description_file, args.image_folder)
    train_generator, test_generator = prepare_dataset(pokedex)
    # Załaduj zapisany model
    model = load_model('model')

    # Przewidywania na zbiorze testowym
    y_true = test_generator.classes  # Prawdziwe klasy (etikiety) z generatora testowego
    y_pred = model.predict(test_generator)  # Przewidywania modelu

    # Zamiana wyników przewidywań na indeksy klas
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Raport klasyfikacji (precision, recall, f1-score)

    class_names = list(test_generator.class_indices.keys())
    print(classification_report(y_true, y_pred_classes, target_names=class_names[:len(np.unique(y_true))]))

    # Macierz pomyłek

    cm = confusion_matrix(y_true, y_pred_classes)

    # Dopasowanie etykiet do liczby klas w zbiorze testowym
    class_names = list(test_generator.class_indices.keys())
    display_labels = class_names[:len(np.unique(y_true))]

    # Wyświetlenie macierzy pomyłek
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical')

    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()