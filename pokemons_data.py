import pandas as pd
import keras as ks
import matplotlib.pyplot as plt
import argparse
import os


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
    parser.add_argument('-c', '--confusion_matrix', default='confustion_matrix.png')
    return parser.parse_args()

def prepare_network():
    pass
def main():
    args = parse_arguments()
    pokedex = load_pokedex(args.description_file, args.image_folder)
    show_dataset_info(pokedex)
    train_generator, test_generator = prepare_dataset(pokedex)
    model = prepare_network() # TODO : here construct the network adding layers
    history = model.fit(train_generator, validation_data=test_generator, epochs=30)
    model.save(args.model_folder)
    
    # TODO : plotting

    if args.history_image:
        plt.savefig(args.history_image)
    else:
        plt.show()


if __name__=='__main__':
    main()
