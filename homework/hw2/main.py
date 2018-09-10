from io_tools import load_data
import math
from model import ConvolutionalNeuralNetwork

def main():
    """High level pipeline."""

    # Load dataset.
    data = load_data("data/MNISTdata.hdf5")
    train_data, test_data = data[:2], data[2:]
    input_dim = int(math.sqrt(train_data[0].shape[1]))  # 28.

    # Init model.
    cnn = ConvolutionalNeuralNetwork(input_dim)

    # Train.
    print("training model...")
    cnn.train(train_data, learning_rate=0.1, epochs=25)

    # Test
    print("testing model...")
    cnn.test(test_data)

if __name__ == "__main__":
    main()
