import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import misc
from tqdm import tqdm
import seaborn as sns
import pandas as pd


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


#0. Choose a corticial layer (i.e. run in data_cifs or not)
exp_name = 'l23'
main_dir = '/home/drew/Documents/contextual_circuit_bp/tests/ALLEN_files/data'
data_dirs = [
    'conv2d',
    'sep_conv2d',
    'DoG'
]
every_nth = 1

#1. Load gradient images from each model
grad_ims = {}
for d in tqdm(data_dirs):
    itd = np.load(os.path.join(main_dir, d, 'data.npz'))
    grad_ims[d] = itd['grads']
    real_ims = itd['images']

#3. Correlate the three grad images across time, producing three dissimilarity lines. Conv-sep, conv-dog, dog-sep.
corrs = []
for t in tqdm(range(grad_ims[d].shape[0])):
    it_mat = []
    for k in grad_ims.keys():
        it_mat += [grad_ims[k][t].ravel()]
    corrs += [np.corrcoef(np.asarray(it_mat))]
corrs = np.asarray(corrs)
corrs[np.isnan(corrs)] = 0.
dog_conv = corrs[:, 0, 2]
dog_sep = corrs[:, 1, 2]
conv_sep = corrs[:, 0, 1]

#4. Plot the lines and add grad image inlays to show when the three models change in their explanations of images.
gr = np.arange(3).repeat(len(dog_conv))
idx = np.arange(len(dog_conv)).reshape(1, -1).repeat(3, axis=0).reshape(1, -1)
df = pd.DataFrame(
    np.vstack((np.hstack((dog_conv, dog_sep, conv_sep)), idx, gr)).transpose(),
    columns=['data', 'idx', 'group'])
np.save(os.path.join(main_dir, 'grad_traces_%s' % exp_name), df)
# sns.lmplot(x='idx', y='data', data=df, hue='group', fit_reg=False)
sns.set_context("paper")
sns.set_style("white")
# sns.set_style("ticks")
sns.lmplot(x='idx', y='data', data=df[df['group'] == 2], fit_reg=False, truncate=True)
plt.savefig(os.path.join(main_dir, 'grad_traces_%s.pdf' % exp_name))
plt.show()


# best_im = np.where(df[df['group'] == 2].data == df[df['group'] == 2].data.max())[0]
best_im = np.where(df[df['group'] == 2].data == np.sort(df[df['group'] == 2].data)[::-1][10])[0]

worst_im = np.where(df[df['group'] == 2].data == df[df['group'] == 2].data.min())[0]
if len(worst_im) > 0:
    worst_im = worst_im[0]
print 'Best is %s' % best_im
print 'Worst is %s' % worst_im


#1. Load gradient images from each model
for d in tqdm(data_dirs):
    #2. Save every nth grad image into a folder
    new_dir = os.path.join(main_dir, d, 'grad_ims')
    make_dir(new_dir)
    grad_res = grad_ims[d][best_im].squeeze()
    # grad_res = (grad_res - grad_res.min()) / (grad_res.max() - grad_res.min())
    grad_res = (grad_res - grad_res.mean()) / grad_res.std()
    real_res = real_ims[best_im].squeeze()  # [:, :, None]
    real_res = (real_res - real_res.mean()) / real_res.std()
    # real_res /= 255.
    # real_res = np.concatenate((real_res, real_res, real_res), axis=-1)
    # real_res = np.concatenate((real_res, grad_res), axis=-1)
    fig, ax = plt.subplots()
    plt.imshow(grad_res * real_res, cmap='inferno')
    # plt.imshow(real_res, cmap='inferno')

    fig.patch.set_visible(False)
    ax.axis('off')
    plt.savefig(os.path.join(new_dir, 'best_combo_%s.png' % best_im))
    plt.close('fig')
    plt.imshow(grad_ims[d][best_im].squeeze(), cmap='Reds')
    plt.savefig(os.path.join(new_dir, 'best_grad_%s.png' % best_im))
    misc.imsave(os.path.join(new_dir, 'best_real_%s.png' % best_im), real_ims[best_im].squeeze())

    grad_res = grad_ims[d][worst_im].squeeze()
    grad_res = (grad_res - grad_res.mean()) / grad_res.std()
    real_res = real_ims[worst_im].squeeze()  # [:, :, None]
    real_res = (real_res - real_res.mean()) / real_res.std()
    # real_res = np.concatenate((real_res, real_res, real_res), axis=-1)
    fig, ax = plt.subplots()
    plt.imshow(grad_res * real_res, cmap='inferno')
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.savefig(os.path.join(new_dir, 'worst_combo_%s.png' % worst_im))
    plt.close('fig')
    plt.imshow(grad_ims[d][worst_im].squeeze(), cmap='Reds')
    plt.savefig(os.path.join(new_dir, 'worst_grad_%s.png' % worst_im))
    misc.imsave(os.path.join(new_dir, 'worst_real_%s.png' % worst_im), real_ims[worst_im].squeeze())


