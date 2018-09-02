from model import NeuralNetwork
from io_tools import load_data

def main():
    """High level pipeline."""

    # Load dataset.
    data = load_data("data/MNISTdata.hdf5")
    train_data, test_data = data[:2], data[2:]

    # Init model.
    nn = NeuralNetwork(train_data[0].shape[1], hidden_units=100)

    # Train.
    print("training model...")
    nn.train(train_data, learning_rate=2.1, epochs=40)

    # Test
    print("testing model...")
    nn.test(test_data)

if __name__ == "__main__":
    main()