"""Plot a single neuron on the allen institute movie data."""
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_data_from_timestamp(d):
    """Pop data from the timestep dict."""
    return [d[idx] for idx in range(len(d.keys()))]


def plot_data(file_list, output_name, title, y='Pearson dissimilarity'):
    """Plot from contextual_bp data."""
    data_files = {}
    for k, v in file_list.iteritems():
        data = np.load(v)
        data_files[k] = get_data_from_timestamp(data.item())

    # Put data in a df
    all_data = np.concatenate((data_files.values()), axis=0)
    id_mat = np.concatenate((
        np.zeros((len(data_files[k]))),
        np.zeros((len(data_files[k]))) + 1,
        np.zeros((len(data_files[k]))) + 2), axis=0)
    time = np.concatenate((
        np.arange(len(data_files[k])),
        np.arange(len(data_files[k])),
        np.arange(len(data_files[k]))), axis=0)
    df = pd.DataFrame(
        np.vstack((all_data, id_mat, time)).transpose(),
        columns=[y, 'model', 'Iteration'])

    # Plot data
    matplotlib.style.use('ggplot')
    plt.rc('font', size=6)
    f, ax = plt.subplots(1, figsize=(10, 10))
    for k, rk in zip(df['model'].unique(), data_files.keys()):
        tmp = df[df['model'] == k]
        ax = tmp.plot(
            x='Iteration',
            y='Pearson dissimilarity',
            label=rk,
            kind='line',
            ax=ax,
            logy=False,
            alpha=0.9)
    plt.legend(prop={'size': 12}, labelspacing=1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    plt.savefig(output_name)
    plt.close(f)


def zscore(x):
    """Standardize vector to 0 mean unit variance."""
    return (x - x.mean()) / x.std()


def standardize_batches(data, bs):
    """Standardize data by batch-sized chunks."""
    res_data = data[:bs * (len(data) // bs)]
    res_data = res_data.reshape(bs, -1)
    out_data = np.zeros((res_data.shape))
    for idx in np.arange(res_data.shape[1]):
        out_data[:, idx] = zscore(res_data[:, idx])
    return out_data.reshape(-1)


def plot_traces(in_dict, title):
    """Plot standardized traces."""
    # pred = standardize_batches(pred, in_dict['batch_size'])
    # gt = standardize_batches(gt, in_dict['batch_size'])
    pred = zscore(np.load(in_dict['pred']).item()[0][-in_dict['batch_size']:])
    gt = zscore(np.load(in_dict['gt']).item()[0][-in_dict['batch_size']:])
    matplotlib.style.use('ggplot')
    plt.rc('font', size=6)
    f, ax = plt.subplots(1, figsize=(20, 10))
    plt.plot(pred, marker='o')
    plt.plot(gt, marker='o')
    plt.legend(prop={'size': 12}, labelspacing=1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Normalized fluorescence.', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    rho = np.around(np.corrcoef(pred, gt)[0, 1], 2)
    ax.set_title('%s rho=%s' % (title.replace('_', ' '), rho), fontsize=12)
    plt.savefig(in_dict['output'])
    plt.close(f)


def main():
    """Run plotting."""

    # Figure 1. Effectivenss of conv. vs. DoGs for fitting Allen cell.
    file_list = {
        'dog': 'ALLEN_all_neurons_dog_train_losses.npy',
        'conv2d': 'ALLEN_all_neurons_conv2d_train_losses.npy',
        'sparse_conv2d': 'ALLEN_all_neurons_sparse_conv2d_train_losses.npy'
    }
    output_name = 'training_losses.tiff'
    title = 'Neuron activity elicited by natural movie.'
    plot_data(file_list, output_name, title)

    # Figure 2. Trace of neural activity on validation data
    bs = 64
    file_list = {
        'DoG': {
            'pred': 'ALLEN_all_neurons_dog_val_scores.npy',
            'gt': 'ALLEN_all_neurons_dog_val_labels.npy',
            'batch_size': bs,
            'output': 'DoG_validation_responses.tiff'
        },
        'conv2d': {
            'pred': 'ALLEN_all_neurons_conv2d_val_scores.npy',
            'gt': 'ALLEN_all_neurons_conv2d_val_labels.npy',
            'batch_size': bs,
            'output': 'conv2d_validation_responses.tiff'
        },
        'sparse_conv2d': {
            'pred': 'ALLEN_all_neurons_sparse_conv2d_val_scores.npy',
            'gt': 'ALLEN_all_neurons_sparse_conv2d_val_labels.npy',
            'batch_size': bs,
            'output': 'sparse_conv2d_validation_responses.tiff'
        }
    }
    for k, v in file_list.iteritems():
        plot_traces(v, title=k)
