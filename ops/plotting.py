import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob


def plot_data(
        train_loss,
        val_loss,
        model_name,
        timestepps,
        config,
        output,
        output_ext,
        data_type):
    """Find all loss files in the output folder and
    plot them vs. current performance."""
    other_files = glob(os.path.join(
        output,
        '*%s*' % data_type)
    )
    f = plt.figure()
    handles = []
    for f in other_files:
        data = np.load(f)
        file_name = f.strip('.npy')
        handles += [plt.plot(data, label=file_name)]
    handles += [plt.plot(
        train_loss, label='%s train (Current model)' % model_name)]
    handles += [plt.plot(
        val_loss, label='%s val (Current model)' % model_name)]
    plt.legend(handles=handles)
    plt.savefig(os.path.join(
        output,
        '%s_%s%s' % (model_name, data_type, output_ext)))
