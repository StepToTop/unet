from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        color_mode="grayscale",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        color_mode="grayscale",
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        seed=seed)
    # return image_generator
    train_generator = zip(image_generator, mask_generator)
    # exit()
    for (img, mask) in train_generator:
        yield img, mask


def testGenerator(test_path, num_image=30):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=True)
        img = img / 255
        img = trans.resize(img, (256, 256))
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), item[:, :, 0])
