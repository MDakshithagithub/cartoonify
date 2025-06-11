import os
import cv2
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf # This might be an alternative way users tried
# tf.disable_v2_behavior() # This might be an alternative way users tried

# Ensure TF1 compatibility
tf.compat.v1.disable_v2_behavior()

import network
import guided_filter
from tqdm import tqdm


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)

    h, w = (h//8)*8, (w//8)*8

    # Add check for zero dimensions
    if h == 0:
        h = 8 # Set to a minimum dimension if it became zero
    if w == 0:
        w = 8 # Set to a minimum dimension if it became zero

    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.compat.v1.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.compat.v1.train.Saver(var_list=gene_vars)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)

            # Add a check if image is None after imread, as cv2.imread might not raise an error for bad paths/files
            if image is None:
                print(f'Failed to read image: {load_path}')
                continue # Skip to the next image

            image = resize_crop(image) # This could also be a source of cv2.error or other errors

            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except cv2.error as e:
            print(f'OpenCV error processing {load_path}: {e}')
        except FileNotFoundError: # Though cv2.imread usually returns None instead of raising this.
            print(f'File not found: {load_path}')
        except Exception as e:
            print(f'An unexpected error occurred with {load_path}: {e}')




if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    os.makedirs(save_folder, exist_ok=True)
    cartoonize(load_folder, save_folder, model_path)
