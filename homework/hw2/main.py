from io_tools import load_data
import math
from model import ConvolutionalNeuralNetwork

DEBUG = 1

def main():
    """High level pipeline."""

    # Load dataset.
    print("loading dataset...")
    x_train, y_train, x_test, y_test =  load_data("data/MNISTdata.hdf5")

    if DEBUG:
        print("running in debug mode...")
        # Take a small amount of data to speed up the debug process.
        x_train = x_train[:10001]
        y_train = y_train[:10001]
        x_test = x_test[:5001]
        y_test = y_test[:5001]

    input_dim = int(math.sqrt(x_train.shape[1]))  # 28.

    # Init model.
    cnn = ConvolutionalNeuralNetwork(input_dim)

    # Train.
    print("training model...")
    cnn.train(x_train, y_train, learning_rate=0.0001, epochs=25)

    # Test
    print("testing model...")
    cnn.test(x_test, y_test)

if __name__ == "__main__":
    main()
