import os
import numpy as np
from db import db
from db import credentials
from config import Config
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
from main import get_dt_stamp
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.plotly as py


def main(
        experiment_name,
        im_ext='.pdf',
        log_transform_loss=True,
        colors='Greens_r',
        exclude=None):
    """Plot results of provided experiment name."""
    config = Config()
    pl_creds = credentials.plotly_credentials()
    py.sign_in(
        pl_creds['username'],
        pl_creds['api_key'])

    # Get experiment data
    perf = db.get_performance(experiment_name=experiment_name)
    if len(perf) == 0:
        raise RuntimeError('Could not find any results.')
    structure_names = [x['model_struct'].split('/')[-1] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    lrs = [x['lr'] for x in perf]
    datasets = [x['dataset'] for x in perf]
    loss_funs = [x['loss_function'] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    wd_types = [x['regularization_type'] for x in perf]
    wd_penalties = [x['regularization_strength'] for x in perf]
    steps = [float(x['training_step']) for x in perf]
    training_loss = [float(x['training_loss']) for x in perf]
    validation_loss = [float(x['validation_loss']) for x in perf]

    # Pass data into a pandas DF
    model_params = ['%s | %s | %s | %s | %s | %s | %s | %s' % (
        ipa,
        ipb,
        ipc,
        ipd,
        ipe,
        ipf,
        ipg,
        iph) for ipa, ipb, ipc, ipd, ipe, ipf, ipg, iph in zip(
            structure_names,
            optimizers,
            lrs,
            loss_funs,
            optimizers,
            wd_types,
            wd_penalties,
            datasets)]

    # DF and plot
    df = pd.DataFrame(
        np.vstack(
            (
                model_params,
                steps,
                training_loss,
                validation_loss
            )
        ).transpose(),
        columns=[
            'model parameters',
            'training iteration',
            'training loss',
            'validation loss'
            ]
        )
    df['training iteration'] = pd.to_numeric(df['training iteration'], errors='coerce')
    df['training loss'] = pd.to_numeric(df['training loss'], errors='coerce')
    if log_transform_loss:
        loss_label = 'Log loss'
        df['training loss'] = np.log(df['training loss'])
    else:
        loss_label = 'Normalized loss (x / max(x))'
        df['training loss'] /= df.groupby(
            'model parameters')['training loss'].transform(max)
    df['validation loss'] = pd.to_numeric(df['validation loss']) * 100.
    if exclude is not None:
        exclusion_search = df['model parameters'].str.contains(exclude)
        df = df[exclusion_search == False]
        print 'Removed %s rows.' % exclusion_search.sum()

    # Start plotting
    plt.rc('font', size=8)
    f, axs = plt.subplots(2, figsize=(20, 30))
    ax = sns.pointplot(
        x='training iteration',
        y='training loss',
        hue='model parameters',
        ci=None,
        estimator=np.sum,
        data=df,
        ax=axs[0],
        scale=.25,
        pallette=colors)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Training')
    ax.set_ylabel(loss_label)
    ax = sns.pointplot(
        x='training iteration',
        y='validation loss',
        hue='model parameters',
        ci=None,
        estimator=np.sum,
        data=df,
        ax=axs[1],
        scale=.25,
        pallette=colors)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Validation')
    ax.set_ylabel('Categorization accuracy (%)')
    out_name = os.path.join(
        config.plots,
        '%s_%s%s' % (
            experiment_name, get_dt_stamp(), im_ext))
    plt.savefig(out_name)
    print 'Saved to: %s' % out_name
    plot_url = py.plot_mpl(f, auto_open=False)
    print 'Uploaded to: %s' % plot_url
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--exclude',
        dest='exclude',
        type=str,
        default=None,
        help='Experiment exclusion keyword.')
    args = parser.parse_args()
    main(**vars(args))
