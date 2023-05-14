import gzip
import os
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from ctgan.synthesizers import CTGAN, TVAE

def load_fake_mnist(path="dataset/mnist/", synthesizer="ctgan", num_epochs=30, pretrained=""):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if nonexistant. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.

    https://mattpetersen.github.io/load-mnist-with-numpy
    Load from /home/USER/data/mnist or elsewhere; download if missing.
    """

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
    ]

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        # return pixels.reshape(-1, 784).astype('float32') / 255
        return (pixels.reshape(-1, 784) != 0).astype(np.int8)

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    train_data = np.concatenate((train_images, train_labels.argmax(-1)[:,None]), axis=1)
    if synthesizer == "ctgan":
        if pretrained:
            model = CTGAN.load(pretrained)
        else:
            model = CTGAN(batch_size=10**3, epochs=num_epochs)
    elif synthesizer == "tvae":
        if pretrained:
            model = TVAE.load(pretrained)
        else:
            model = TVAE(batch_size=10**3, epochs=num_epochs)
    else:
        print(f"Synthesizer {synthesizer} not defined!")
        return
    if not pretrained:
        model.fit(train_data, discrete_columns=list(range(train_data.shape[1])), epochs=num_epochs)
        # Training fake data synthesizer for MNIST takes a lot of time, save the model to reuse it
        now = datetime.datetime.now()
        current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
        model.save(f"../models/mnist_{synthesizer}_{num_epochs}_epochs_{current_time}.pkl")
    fake_data = model.sample(train_data.shape[0])

    mnist_fake_train_X, mnist_fake_valid_X, mnist_fake_train_y, mnist_fake_valid_y = train_test_split(fake_data[:,:-1], np.eye(10)[fake_data[:,-1]], test_size=0.1, random_state=1, shuffle=True, stratify=fake_data[:,-1])

    return mnist_fake_train_X, mnist_fake_train_y, mnist_fake_valid_X, mnist_fake_valid_y

mnist_fake_train_X, mnist_fake_train_y, mnist_fake_valid_X, mnist_fake_valid_y = load_fake_mnist("../dataset/mnist/", synthesizer="ctgan", num_epochs=30)