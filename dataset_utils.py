import os
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def carrega_dataset(base_dir, batch_size):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=False)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary', shuffle=False)

    return train_generator, validation_generator, test_generator