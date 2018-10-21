import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import argparse
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

def plot_query_results():
    """Plot top 10 and bottom 10 ranking results
    Args:
        path(str): path to the file that stores the ranking results
    File format:
        idx=0, query_path label 0
        idx=[1, 10], top10_path label distances
        idx=[11, 20], bottom10_path label distances
    """

    for file_idx in range(1, 6):
        with open('query_results_' + str(file_idx) + '.txt', 'r') as file:
            # Parse results.
            top_paths, top_labels, top_dists = [], [], []
            bot_paths, bot_labels, bot_dists = [], [], []

            for i, line in enumerate(file):
                line = line.strip().split()
                if i == 0:
                    q_path, q_label, dist0 = line
                    top_paths.append(q_path)
                    top_labels.append(q_label)
                    top_dists.append(dist0)
                    bot_paths.append(q_path)
                    bot_labels.append(q_label)
                    bot_dists.append(dist0)
                elif 1 <= i <= 10:
                    top_paths.append(line[0])
                    top_labels.append(line[1])
                    top_dists.append(line[2])
                else:
                    bot_paths.append(line[0])
                    bot_labels.append(line[1])
                    bot_dists.append(line[2])
            
            # Plot top 10.
            sub_plot(top_paths, top_labels, top_dists, 
                     'query_plot_{}_top.png'.format(file_idx))
            # Plot bottom 10.
            sub_plot(bot_paths, bot_labels, bot_dists, 
                     'query_plot_{}_bottom.png'.format(file_idx))

def sub_plot(paths, labels, distances, save_path):
    fig = plt.figure()
    
    total = len(paths)
    for i in range(total):
        image = Image.open(paths[i])
        image = np.array(image.convert('RGB'))
        label = labels[i]
        dist = distances[i]
        
        ax = plt.subplot(4, 3, i + 1)
        ax.set_title("{}\n{}".format(label, dist))
        ax.title.set_fontsize(3)
        ax.axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.imshow(image)
        plt.savefig(save_path, format="png", dpi=600)


if __name__ == '__main__':
    if args.plot_type == 'loss':
        plot_train_loss(args.path)
    else:
        plot_query_results()