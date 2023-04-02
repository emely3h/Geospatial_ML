import tensorflow as tf
import matplotlib.pyplot as plt


def _display_image(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if len(display_list[i].shape) == 3:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            # plt.axis('off')
        else:
            plt.imshow(display_list[i])
    plt.show()


def display(list_input, list_mask):
    """
    Plots multiple sets of input tile and mask tile

    Args:
        list_input: np.array holding multiple input tiles of shape (x, 256, 256, 5)
        list_mask: np.array holding multiple mask tiles of shape (x, 256, 256)
    """

    for idx, img_train in enumerate(list_input):
        sample_image, sample_mask = list_input[idx], list_mask[idx]
        sample_image = sample_image[..., :4]
        _display_image([sample_image, sample_mask])
