import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate plots")
parser.add_argument("--plot_type", default='loss', const='loss', nargs='?', 
                    choices=['loss', 'query_result'],
                    help="type of the plot to generate")
parser.add_argument("--path", type=str, 
                    help="path of data file from which to generate the plot")

args = parser.parse_args()


def plot_train_loss(path):
    """Plot loss during training
    Args:
        path(str): Bluewaters' .out file name
    """

    # Parse file to get loss.
    loss = []
    steps_per_epoch = 0
    total_steps = 0

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line[:8] != 'Training':
                continue
            
            line = line.split()
            loss.append(float(line[-1]))
            
            if line[2][:-1] == '1':
                steps_per_epoch += 1
            total_steps += 1
    
    num_epochs = int(line[2][:-1])

    # Convert steps to epochs on the figure.
    x = list(range(1, total_steps + 1))
    x_steps = list(range(1, total_steps+1, steps_per_epoch))
    x_epochs = list(range(1, num_epochs+1))

    # Plot.
    plt.plot(x, loss, linestyle='-', clip_on=False, color='PaleVioletRed')

    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.xticks(x_steps, x_epochs)

    plt.savefig("train_loss.png", format='png', dpi=300)

def plot_query_results(path):
    """Plot top 10 and bottom 10 ranking results
    Args:
        path(str): path to the file that stores the ranking results
    File format:
        idx=0, query_path label 0
        idx=[1, 10], top10_path label distances
        idx=[11, 20], bottom10_path label distances
    """


if __name__ == '__main__':
    if args.plot_type == 'loss':
        plot_train_loss(args.path)
    else:
        plot_query_results(args.path)