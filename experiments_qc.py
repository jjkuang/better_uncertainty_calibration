import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from matplotlib import rc
from scipy.special import softmax
from scipy.stats import entropy
from pathos.multiprocessing import ProcessingPool as Pool

from temperature_scaling import TemperatureScaling, ETScaling, fit_logits_scale, save_aupr_logits
from errors import BS, sECE, get_acc_conf_bins, NLL, ECE, rBS, get_imagenet_a_results, get_imagenet_o_results, get_acc_vs_conf_thr_results
from utils import logits_RC_to_pkl, unpickle_probs_RC, unpickle_probs_OOD, parse_logits_path, MODELS_MAP, METHODS_MAP, PRECISIONS_MAP

import os
# supposedly makes multithreading quicker, true though?
os.environ["OPENBLAS_MAIN_FREE"] = "1"
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=100)
sns.set_context("paper")
rc('text', usetex=True)

# all errors are squared; we will take the root during plotting
errors = {
    'RBS': rBS,
    # 'KDE CE': sKDECE,
    # 'KS CE': sKSCE,
    # 'd EM 15b TCE': dEMsTCE,
    'ECE': ECE,
    'NLL': NLL,
    'AURRA': get_imagenet_a_results,
    'AUPR': get_imagenet_o_results
    # '100b TCE': lambda l, y: sTCE(l, y, bins=100),
    # '15b CWCE': sCWCE,
    # '100b CWCE': lambda l, y: sCWCE(l, y, bins=100),
}

class Experiments:

    def __init__(self):
        # never have 2 different exp_id with the same exp_seed
        self.columns = ['model', 'dataset', 'precision', 'method', 
                        'RC_probs_path', 'ood_probs_path', 'top1', 'top5', 'nll']
        self.results_df = pd.DataFrame()
        # colors for the plots
        self.palette = {
            'RBS': '#e63946', 
            'ECE': '#FF8800',
            }
        rest_size = 3
        # line sizes for the plots
        self.sizes = {
            'upper bound': 5, 'RBS': 5,
            'ECE': rest_size,
            }

    def save(self, filename):
        self.results_df.to_csv(filename, index=False)

    def load(self, filename, append=True, reset_index=True):
        loaded_df = pd.read_csv(filename)
        if append:
            print("Appending csv df to current Experiments...")
            self.results_df = pd.concat([self.results_df, loaded_df])
        else:
            self.results_df = loaded_df
        if reset_index:
            self.results_df.reset_index(inplace=True, drop=True)

    def reset(self):
        self.results_df = pd.DataFrame(columns=self.columns)

    def _eval_run(
        self, ce_type, ce_func, lgts, y, lgts_RC
    ):
        start_t = time.time()

        v = ce_func(lgts, y)
        v_RC = ce_func(lgts_RC, y)

        return {
            ce_type: v,
            ce_type+'_TS': v_RC
        }

    def _eval_run_adv(
        self, ce_type, ce_func, lgts_iid, lgts_ood
    ):
        start_t = time.time()

        v = ce_func(lgts_iid, lgts_ood)

        return { ce_type: v }

    def add_experiment(
        self, metrics_df, model, method, seed=None, ce_types=None, adv=False
    ):
        # use start time as ID
        exp_id = time.time()

        if seed is None:
            seed = int(exp_id)
        if ce_types is None:
            ce_types = list(errors.keys())

        np.random.seed(seed)
        n_ticks = 10

        def iter_func():
            runs = {}
            def add_run(ce_type, func):
                try:
                    error = self._eval_run(ce_type, func, logits_test, labels_test, logits_test_RC)
                    runs.update(error)
                except AssertionError:
                    pass

            # TODO: how to make sure ImageNet-O gets both runs??
            def add_run_adv(ce_type, func):
                try:
                    error = self._eval_run_adv( 
                        ce_type, func, logits_test, logits_test_RC) # i.e., iid/ood even tho not rly
                    runs.update(error)
                except AssertionError:
                    pass

            for ce_type in ce_types:
                if ce_type == "AUPR":
                    if adv:
                        logits_test, logits_test_RC, labels_test, _ = unpickle_probs_OOD(metrics_df.at[0, 'ood_probs_path'])
                        add_run_adv(ce_type, errors[ce_type])
                        continue
                    else:
                        continue

                logits_test, logits_test_RC, labels_test = unpickle_probs_RC(metrics_df.at[0, 'RC_probs_path'])
                add_run(ce_type, errors[ce_type])

            return pd.DataFrame([runs])

        # pool = Pool()
        # results = pool.map(iter_func, repeated_sizes)
        results = iter_func()
        metrics_df = metrics_df.join(results)
        self.results_df = pd.concat([self.results_df, metrics_df])

    def add_DIAG_experiment(
        self, logits, labels, logits_RC, setting, start_n=1000, start_rep=2500,
        seed=None, ce_types=None
    ):
        # use start time as ID
        exp_id = time.time()

        if seed is None:
            seed = int(exp_id)
        if ce_types is None:
            ce_types = list(errors.keys())

        np.random.seed(seed)
        n_ticks = 10

        n = logits.shape[0]

        sizes = np.rint(np.flip(np.logspace(np.log2(start_n), np.log2(n), n_ticks, base=2))).astype(int)
        # quadratically decrease repetitions
        repetitions = np.rint(np.linspace(1, np.sqrt(start_rep), n_ticks) ** 2).astype(int)

        repeated_sizes = [s for s, r in zip(sizes, repetitions) for _ in range(r)]

        def iter_func(s):
            indices = np.random.choice(n, size=s, replace=False)
            ls = logits[indices]
            ys = labels[indices]
            ls_RC = logits_RC[indices]

            runs = []

            def add_run(ce_type, func):
                try:
                    error = self._eval_run(
                        ce_type, func, ls, ys, ls_RC, s, setting, exp_id, seed,
                        method='DIAG')
                    runs.append(error)
                except AssertionError:
                    pass

            for ce_type in ce_types:
                add_run(ce_type, errors[ce_type])

            return pd.DataFrame(runs)

        pool = Pool()
        results = pool.map(iter_func, repeated_sizes)
        results = pd.concat(results)
        self.results_df = self.results_df.append(results)

    def saving_plot(self, plot, save_file, fmt, tight=True, **kwargs):
        if save_file is not None:
            if tight:
                plot.get_figure().savefig(
                    'plots/{}.{}'.format(save_file, fmt), bbox_inches='tight', **kwargs)
                plt.close()
            else:
                plot.get_figure().savefig(
                    'plots/{}.{}'.format(save_file, fmt), **kwargs)
                plt.close()

    def get_legend(
        self, size=(2.5, 2.5), ce_types=None, save_file=None, font_scale=1.3,
        padding=7
    ):

        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            df = self.results_df[
                self.results_df['setting'] == 'densenet40_c10']
        else:
            df = self.results_df[
                (self.results_df['setting'] == 'densenet40_c10')
                & (self.results_df['ce_type'].isin(ce_types))
            ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        df = df.reset_index(drop=True)

        plot = sns.lineplot(
            data=df, x='test_size', y='ce_value', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        # hide the plot behind the legend
        plot.set(
            xticklabels=[], xlabel=None, ylabel=None, ylim=(-100, 100),
            xlim=(-100000, 100000))
        plot.set_yticks([])
        plot.legend(
            loc='center', framealpha=1, frameon=True, fancybox=True,
            borderpad=padding, title='Calibration Estimator')
        sns.despine(left=True, bottom=True)
        self.saving_plot(plot, save_file, tight=False)

    def plot_CE(
        self, setting, size=(4.5, 3), use_root=True, ce_types=None,
        save_file=None, legend=False, font_scale=1.4
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            df = self.results_df[self.results_df['setting'] == setting]
        else:
            df = self.results_df[
                (self.results_df['setting'] == setting)
                & (self.results_df['ce_type'].isin(ce_types))
            ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        df['root_ce_value'] = df['ce_value'] ** p
        df = df.reset_index(drop=True)

        plot = sns.lineplot(
            data=df, x='test_size', y='root_ce_value', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        ticks = df['test_size'].unique()
        plot.set(
            xscale='log', xticks=ticks.astype(float), xticklabels=ticks,
            ylim=(0, None), xlabel='Test set size', ylabel='Cal. Error')
        plot.grid(axis='y')
        if not legend:
            plot.legend([], [], frameon=False)
        else:
            plot.legend(title='')
        plot.tick_params(axis='x', rotation=45)
        sns.despine()
        self.saving_plot(plot, save_file)

    def plot_RC_delta(
        self, setting, size=(4.5, 3), use_root=True, ce_types=None,
        save_file=None, legend=False, font_scale=1.4
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            df = self.results_df[self.results_df['setting'] == setting]
        else:
            df = self.results_df[
                (self.results_df['setting'] == setting)
                & (self.results_df['ce_type'].isin(ce_types))
            ]
        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        plot = sns.lineplot(
            data=df, x='test_size', y='RC_delta', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        ticks = df['test_size'].unique()
        plot.set(
            xscale='log', xticks=ticks.astype(float), xticklabels=ticks,
            xlabel='Test set size', ylabel='Cal. Improvement')
        if not legend:
            plot.legend([], [], frameon=False)
        else:
            plot.legend(title='')
        plot.grid(axis='y')
        plot.tick_params(axis='x', rotation=45)
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_RC_sbias(
        self, settings, size=(9.2, 7.2), use_root=True, ce_types=None,
        save_file=None
    ):
        """
        self bias
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            df['test_size'].isin([2897, 10000, 25000, 26032])]

        # god have mercy with my cpu
        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                for error in df['ce_type'].unique():
                    rows_delta = (best_deltas['setting'] == setting) & \
                        (best_deltas['RC_method'] == RC_method) & \
                        (best_deltas['ce_type'] == error)
                    best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                    rows_df = (df['setting'] == setting) & \
                        (df['RC_method'] == RC_method) & \
                        (df['ce_type'] == error)
                    df.loc[rows_df, 'RC_self_bias'] = (
                        df.loc[df['setting'] == setting, 'RC_delta'] - best_delta)
        plot = sns.lineplot(
            data=df, x='test_size', y='RC_self_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        plot.grid(axis='y')
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_RC_rsbias(
        self, settings, size=(9.2, 7.2), use_root=True, ce_types=None,
        save_file=None
    ):
        """
        relative self bias
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p / df['ce_RC_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            df['test_size'].isin([2897, 10000, 25000, 26032])]

        # god have mercy with my cpu
        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                for error in df['ce_type'].unique():
                    rows_delta = (best_deltas['setting'] == setting) & \
                        (best_deltas['RC_method'] == RC_method) & \
                        (best_deltas['ce_type'] == error)
                    best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                    rows_df = (df['setting'] == setting) & \
                        (df['RC_method'] == RC_method) & \
                        (df['ce_type'] == error)
                    df.loc[rows_df, 'RC_bias'] = (
                        df.loc[df['setting'] == setting, 'RC_delta'] - best_delta)

        plot = sns.lineplot(
            data=df, x='test_size', y='RC_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        plot.grid(axis='y')
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_rsbias(
        self, settings, size=(9.2, 7.2), use_root=True, ce_types=None,
        save_file=None, legend=False, font_scale=1.15
    ):
        """
        relative self bias
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            df['test_size'].isin([2897, 10000, 25000, 26032])]

        # god have mercy with my cpu
        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                for error in df['ce_type'].unique():
                    rows_delta = (best_deltas['setting'] == setting) & \
                        (best_deltas['RC_method'] == RC_method) & \
                        (best_deltas['ce_type'] == error)
                    best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                    rows_df = (df['setting'] == setting) & \
                        (df['RC_method'] == RC_method) & \
                        (df['ce_type'] == error)
                    df.loc[rows_df, 'relative_self_bias'] = (
                        df.loc[df['setting'] == setting, 'RC_delta'] / best_delta)

        plot = sns.lineplot(
            data=df, x='test_size', y='relative_self_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        if not legend:
            plot.legend([], [], frameon=False)
        else:
            plot.legend(title='')
        ticks = df['test_size'].unique()
        plot.set(
            xscale='log', xticks=ticks.astype(float), xticklabels=ticks,
            xlabel='Test set size', ylabel='Relative bias')
        plot.grid(axis='y')
        plot.tick_params(axis='x', rotation=45)
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_RC_bias(
        self, settings, size=(9.2, 7.2), use_root=False, ce_types=None,
        save_file=None
    ):
        """
        save_file: Str, no parent folders or suffix (file type)
        """
        # use the square root is actually false here as this is not the bias
        # anymore
        p = 1 # .5 if use_root else 1
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        # Cal. Improvement of RC in the squared space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            (df['test_size'].isin([2897, 10000, 25000, 26032]))
            & (df['ce_type'] == 'BS')]

        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                rows_delta = (best_deltas['setting'] == setting) & (best_deltas['RC_method'] == RC_method)
                best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                rows_df = (df['setting'] == setting) & (df['RC_method'] == RC_method)
                df.loc[rows_df, 'RC_bias'] = (
                    df.loc[df['setting'] == setting, 'RC_delta'] - best_delta)

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.lineplot(
            data=df, x='test_size', y='RC_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        plot.grid(axis='y')
        sns.despine()
        self.saving_plot(plot, save_file)

    def plot_runtime(
        self, setting, size=(12.8, 9.6), ce_types=None, log_y=False,
        save_file=None
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})

        if ce_types is None:
            df = self.results_df[self.results_df['setting'] == setting]
        else:
            df = self.results_df[
                (self.results_df['setting'] == setting)
                & (self.results_df['ce_type'].isin(ce_types))
            ]
        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.lineplot(
            data=df, x='test_size', y='runtime', hue='ce_type')
        if log_y:
            plot.set(yscale='log')
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        self.saving_plot(plot, save_file)

    def boxplot(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None, set_size='high'
    ):

        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")
        # not my proudest moment...
        if (ce_types is None) and (settings is None):
            df = self.results_df
        elif (ce_types is None) and (settings is not None):
            df = self.results_df[
                (self.results_df['setting'].isin(settings))
            ]
        elif ce_types is not None and settings is None:
            df = self.results_df[
                (self.results_df['ce_type'].isin(ce_types))
            ]
        else:
            df = self.results_df[
                (self.results_df['ce_type'].isin(ce_types))
                & (self.results_df['setting'].isin(settings))
            ]
        p = .5 if use_root else 1
        df['root_ce_value'] = df['ce_value'] ** p
        # this only works cause the ticks differ per dataset
        if set_size == 'max':
            df = df[df['test_size'].isin([2897, 10000, 25000, 26032])]
        elif set_size == 'high':
            df = df[df['test_size'].isin([1993, 5995, 13536, 14032])]
        elif set_size == 'min':
            df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='root_ce_value', x='setting', hue='ce_type')
        plot.set(ylim=(0, 0.8))
        sns.despine()
        self.saving_plot(plot, save_file)

    def boxplot_RC_delta(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None, set_size='high'
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """

        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")
        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        p = .5 if use_root else 1
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        # this only works cause the ticks differ per dataset
        if set_size == 'max':
            df = df[df['test_size'].isin([2897, 10000, 25000, 26032])]
        elif set_size == 'high':
            df = df[df['test_size'].isin([1993, 5995, 13536, 14032])]
        elif set_size == 'min':
            df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='RC_delta', x='setting', hue='ce_type')
        sns.despine()
        self.saving_plot(plot, save_file)

    def boxplot_delta(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """

        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")
        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]
        p = .5 if use_root else 1
        df['root_ce_value'] = df['ce_value'] ** p
        df['size_delta'] = df['ce_value'] ** p
        # this only works cause the ticks differ per dataset
        for set in df['setting'].unique():
            for typ in df['ce_type'].unique():
                # value to substract
                val = df[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'].isin([2897, 10000, 25000, 26032]))
                    ]['root_ce_value'].iloc[0]
                # column to substract from
                col = df[
                    (df['setting'] == set)
                    & (df['test_size'] == 100)
                    & (df['ce_type'] == typ)
                    ]['root_ce_value']
                df.loc[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'] == 100),
                    'size_delta'] = col - val
        df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='size_delta', x='setting', hue='ce_type')
        sns.despine()
        self.saving_plot(plot, save_file)

    def boxplot_delta_RC_delta(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """

        sns.set(rc={'figure.figsize': size})
        sns.set_style("white")
        # not my proudest moment...
        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]
        p = .5 if use_root else 1
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        df['size_delta'] = df['RC_delta']
        # this only works cause the ticks differ per dataset
        for set in df['setting'].unique():
            for typ in df['ce_type'].unique():
                # value to substract
                val = df[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'].isin([2897, 10000, 25000, 26032]))
                    ]['RC_delta'].iloc[0]
                # column to substract from
                col = df[
                    (df['setting'] == set)
                    & (df['test_size'] == 100)
                    & (df['ce_type'] == typ)
                    ]['RC_delta']
                df.loc[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'] == 100),
                    'size_delta'] = col - val
        df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='size_delta', x='setting', hue='ce_type')
        sns.despine()
        self.saving_plot(plot, save_file)

    # Adapted from
    # https://github.com/Guoxoug/PTQ-acc-cal/blob/7892fec046615032613148a2716d0066837dda24/utils/plot_utils.py
    # https://github.com/google-research/robustness_metrics/tree/master/robustness_metrics/projects/revisiting_calibration
    def plot_reliability_curve(
        self, settings, save_file=None, size=(7.2, 8.6), save_fmt='pdf', n_bins=15, histogram=True 
    ):
        """Plot a reliability curve given logits and labels.
        
        Also can optionally have a histogram of confidences.
        """
        # Labels should be loaded from the "golden" indices
        # Group by settings to get all probs_paths
        settings = self._validate_dataset_settings(settings)

        df = self.results_df[
                                    (self.results_df['method'].isin(settings['method']))
                                &   (self.results_df['dataset'].isin(settings['dataset']))
                            ]
        df['method'] = df['method'].replace(METHODS_MAP)
        df = self._duplicate_fp_results(df, ['fq-vit'])

        fig, axs = plt.subplots(len(settings['precision']), len(settings['model']), figsize=size)
        set_xlbl, set_ylbl1, set_ylbl2 = False, False, False

        for i, precision in enumerate(settings['precision']): # by row
            set_xlbl = (i == len(settings['precision']) - 1) 
            set_title = (i == 0)
            for j, model in enumerate(settings['model']): # by col
                set_ylbl1 = (j == 0)
                set_ylbl2 = (j == len(settings['model']) - 1)

                ax = axs[i,j]
                ax.plot(
                    [0, 1], [0, 1],
                    linestyle='--', color="black", label="perfect calibration"
                )

                # Current df slice
                df_slice = df[
                                (df['precision'] == precision) & (df['model'] == model)
                            ].reset_index(drop=True)

                print(f"{model}, {precision}")
                assert len(df_slice.index) == 1, f"Found multiple valid slices in df,\n{df_slice}"
                
                logits_test, logits_test_RC, labels = unpickle_probs_RC(df_slice.at[0, 'RC_probs_path'])

                logits_plot = logits_test if settings['logits_RC'] == 0 else logits_test_RC
                probs = softmax(logits_plot, axis=1) 
                acc_in_bin, conf_in_bin, all_confs = get_acc_conf_bins(
                                                        probs, 
                                                        labels, 
                                                        debias=False, 
                                                        num_bins=n_bins, 
                                                        mode='top-label'
                                                    )

                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_widths = bin_boundaries[1:] - bin_boundaries[:-1]
                bin_centers = bin_boundaries[:-1] + (bin_widths / 2)

                ax.bar(
                    bin_centers, 
                    height=acc_in_bin, 
                    width=bin_widths,
                    color="royalblue", 
                    edgecolor="k",
                    linewidth=0.5,
                    alpha=0.6
                )
                ax.bar(
                    bin_centers,
                    bottom=acc_in_bin,
                    height=np.array(conf_in_bin) - np.array(acc_in_bin),
                    width=bin_widths,
                    facecolor=(1, 0, 0, 0.3),
                    edgecolor=(1, 0.3, 0.3),
                    linewidth=0.5
                )
                           
                ax.grid(True, which='major')
                ax.set_xlim(0.0,1.0); ax.set_ylim(0.0,1.0)
                ax.tick_params(axis='both', labelsize=6)

                if set_xlbl:
                    scaled = "(temp-scaled)" if settings['logits_RC'] == 1 else "(unscaled)"
                    ax.set_xlabel(f"Confidence {scaled}")
                else:
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_xlabel(None)
                
                if set_ylbl1:
                    ax.set_ylabel("Accuracy", color="indianred")
                    prec_annot = precision.split('_')
                    precision_label = f"w{prec_annot[0]}a{prec_annot[1]}att{prec_annot[2]}"
                    ax.annotate(f"{precision_label}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                                xycoords=ax.yaxis.label, textcoords='offset points',
                                ha='right', va='center', fontsize=12, rotation=90)
                else:
                    ax.set_ylabel(None)
                    ax.get_yaxis().set_ticklabels([])


                if set_title:
                    ax.set_title(f"{MODELS_MAP[model]}")

                # histogram shows density of predictions wrt confidence
                if histogram:
                    ax2 = ax.twinx()
                    ax2.hist(
                        all_confs,
                        density=True, # density = counts / (sum(counts) * np.diff(bins))
                        bins=n_bins,
                        color="darkcyan",
                        alpha=0.3,
                        hatch='/',
                        range=(0,1)
                    )
                    if set_ylbl2:
                        ax2.set_ylabel("Density", color="darkcyan")
                    ax2.grid(False)
                    ax2.tick_params(axis='y', labelsize=6)

        scaled = "temp-scaled" if settings['logits_RC'] == 1 else "unscaled"
        method = df[df['method'] != 'float']['method'].unique()[0]
        fig_title = fig.suptitle(f"{settings['dataset'][0]} ({method}, {scaled})", y=0.96)
        plt.subplots_adjust(hspace=0.2, wspace=0.3)
        kwargs = {'bbox_extra_artists': [fig_title]}
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True) #, **kwargs)

    def plot_pareto_acc_vs_uncertainty(
        self, settings, save_file=None, size=(9,6), save_fmt='pdf',
        x_lim=[0,1.0], ece_lim=[0,1.0], rbs_lim=[0,1.0]
    ):
        """
        Plots a pareto chart for ECE and RBS vs top-1 error PER quant method
        Models (settings) are shapes, precision is size
        """

        # settings = self._validate_dataset_settings(settings)

        assert len(settings['method']) <= len(METHODS_MAP), f"Only expect at most {len(METHODS_MAP)} methods specified"

        fig, ax = plt.subplots(2, 2, figsize=size)

        df = self.results_df[
                              (self.results_df['model'].isin(settings['model']))
                            & (self.results_df['dataset'].isin(settings['dataset']))
                            & (self.results_df['precision'].isin(settings['precision']))
                            & (self.results_df['method'].isin(settings['method']))
                            ]

        # import pdb; pdb.set_trace()
        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=(PRECISIONS_MAP[k] for k in settings['precision'] if k in PRECISIONS_MAP),
                                         ordered=True)

        df['top1'] = df['top1'].apply(lambda x: 1.0 - (x/100.0))

        df['precision'] = df['precision'].cat.remove_unused_categories()

        n_hue = sorted(df['model'].unique())
        palette = dict(zip(n_hue, sns.color_palette("Set2", n_colors=len(n_hue))))
        markers=['o', 'p', 'D', '*', 'P']
        sizes = (20, 200)
        prec_size_dict = {}
        for sz, prec in zip(np.linspace(sizes[1],sizes[0],len(df['precision'].unique())), df['precision'].cat.categories):
            prec_size_dict[prec] = sz 

        kwargs = dict(
                        sizes=prec_size_dict, #(50, 400), 
                        palette=palette, markers=markers[:len(df['model'].unique())], 
                        edgecolor='k', linewidth=1.0,
                        alpha=0.75, legend='brief'
                    )
        
        sns.scatterplot(
                data=df, x="top1", y="ECE", 
                hue="model", style="model", size="precision", ax=ax[0,0],
                **kwargs
            )
        ax[0,0].set_title("ECE", fontsize=12)


        sns.scatterplot(
                data=df, x="top1", y="ECE_TS", 
                hue="model", style="model", size="precision", ax=ax[0,1],
                **kwargs
            )
        ax[0,1].set_title("ECE, post-TS", fontsize=12)

        sns.scatterplot(
                data=df, x="top1", y="RBS", 
                hue="model", style="model", size="precision", ax=ax[1,0],
                **kwargs
            )
        ax[1,0].set_title("RBS", fontsize=12)

        sns.scatterplot(
                data=df, x="top1", y="RBS_TS", 
                hue="model", style="model", size="precision", ax=ax[1,1],
                **kwargs
            )
        ax[1,1].set_title("RBS, post-TS", fontsize=12)

        for i in range(2):
            for j in range(2):
                ax[i,j].get_legend().remove()
                ax[i,j].set_xlim(x_lim[0],x_lim[1])
                if i == 0:
                    ax[i,j].set_ylim(ece_lim[0],ece_lim[1]) 
                    ax[i,j].set_xlabel(None)
                if i == 1:
                    ax[i,j].set_xlabel("Top-1 ImageNet error")
                    ax[i,j].set_ylim(rbs_lim[0],rbs_lim[1])
                if j == 1:
                    ax[i,j].get_yaxis().set_ticklabels([])
                    ax[i,j].set_ylabel(None)

                ax[i,j].grid(b=True, which='major', color='grey', linestyle='-', alpha=0.1)

        ax[0,0].set_ylabel("ECE")
        ax[1,0].set_ylabel("RBS")

        hdls, lbls = ax[0,1].get_legend_handles_labels()

        for hdl in hdls[1:lbls.index('precision')]:
            hdl.set_sizes([140])

        lbls[0] = r"\textbf{Model}"
        lbls[lbls.index('precision')] = r"\textbf{Precision}"

        lgd = ax[0,1].legend(
            handles=hdls, labels=lbls,
            loc='center right', bbox_to_anchor=(1.4, 0), # 2.4, 0.5), 
            markerscale=0.6 #, ncol=5
        )

        fig.tight_layout()
        method = df[df['method'] != 'float']['method'].unique()[0]
        method_title = fig.suptitle(f"{method}", fontsize=14, x=0.475)
        plt.subplots_adjust(hspace=0.275, wspace=0.115, left=0.1, right=0.84, top=0.9, bottom=0.1)
        kwargs = {'bbox_extra_artists': (lgd,method_title)}
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True, **kwargs)


    def plot_error_and_uncertainty_vs_precision(
        self, settings, save_file=None, save_fmt='pdf', top1_lim=[0,1.0], nll_lim=[0,2.0]
    ):
        """
        Plots a pareto chart for ECE and RBS vs top-1 error
        Models (settings) are shapes, quant method is shade, precision is size
        """

        settings = self._validate_dataset_settings(settings)

        fig, ax = plt.subplots(1, 2, figsize=(8,3))

        df = self.results_df[
                              (self.results_df['model'].isin(settings['model']))
                            & (self.results_df['precision'].isin(settings['precision']))
                            & (self.results_df['method'].isin(settings['method']))
                            ]

        df = self._duplicate_fp_results(df, [*METHODS_MAP])

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=PRECISIONS_MAP.values(), #['FP32', 'w8a8att32', 'w8a8att8', 'w4a4att32', 'w4a4att8'],
                                         ordered=True)

        df['top1'] = df['top1'].apply(lambda x: 1.0 - (x/100.0))


        # Separate model by color
        # Separate method by marker
        n_hue = sorted(df['model'].unique())
        palette = dict(zip(n_hue, sns.color_palette("Set2", n_colors=len(n_hue))))
        markers=['o', 'P', '*', 'p', 'D']

        kwargs = dict( 
                        palette=palette, markers=markers[:len(df['method'].unique())], 
                        markeredgecolor='k', markersize=6, linewidth=2,
                        alpha=0.9, legend='brief'
                    )
        
        sns.lineplot(
                data=df, x="precision", y="top1", 
                hue="model", style="method",  ax=ax[0],
                **kwargs
            )
        ax[0].set_title("Top-1 ImageNet error by precision", fontsize=12)


        sns.lineplot(
                data=df, x="precision", y="NLL", 
                hue="model", style="method", ax=ax[1],
                **kwargs
            )
        ax[1].set_title("NLL by precision", fontsize=12)

        for i in range(2):
            ax[i].get_legend().remove()
            ax[i].grid(b=True, which='major', color='grey', linestyle='-', alpha=0.1)
            ax[i].set_xlabel("Precision")

        ax[0].set_ylim(top1_lim[0], top1_lim[1])
        ax[1].set_ylim(nll_lim[0], nll_lim[1])
        ax[0].set_ylabel("Top-1 ImageNet Error")
        ax[1].set_ylabel("NLL")

        hdls, lbls = ax[1].get_legend_handles_labels()

        lbls[0] = r"\textbf{Model}"
        lbls[lbls.index('method')] = r"\textbf{Method}"

        lgd = ax[1].legend(
            handles=hdls, labels=lbls,
            loc='upper right', bbox_to_anchor=(1.48, 0.855), #2.4, 0.5), 
            markerscale=0.6 #, ncol=5
        )

        plt.subplots_adjust(hspace=0.4, wspace=0.25, left=0.1, right=0.83, top=0.9, bottom=0.15)
        kwargs = {"bbox_extra_artists":(lgd,)}
        if save_file is not None:
            self.saving_plot(fig, save_file, fmt=save_fmt, tight=True, **kwargs)

    def plot_RC_deltas_by_precision(
        self, settings, save_file=None, size=(8,3), save_fmt='pdf', rc_lim=[0,0.5]
    ):
        """
        """
        settings = self._validate_dataset_settings(settings)

        df = self.results_df[
                              (self.results_df['model'].isin(settings['model']))
                            & (self.results_df['dataset'].isin(settings['dataset']))
                            & (self.results_df['precision'].isin(settings['precision']))
                            & (self.results_df['method'].isin(settings['method']))
                            ]

        df = self._duplicate_fp_results(df, [*METHODS_MAP])
        df = df[df['method'].isin(settings['method'])]

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=(PRECISIONS_MAP[k] for k in settings['precision'] if k in PRECISIONS_MAP),
                                         ordered=True)

        fig, ax = plt.subplots(1, len(df['method'].unique()), figsize=size)

        # compute the CE delta values
        # then melt so that we can easily create plot groups by column name
        # ece_delta and rbs_delta go under 'ce_types' and their values become
        # part of another column 'ce_delta'
        df['ece_delta'] = df['ECE'] - df['ECE_TS']
        df['rbs_delta'] = df['RBS'] - df['RBS_TS']
        df = df.drop(columns=['top1', 'top5', 'RC_probs_path', 
                              'ECE', 'ECE_TS', 'RBS', 'RBS_TS', 'NLL', 'NLL_TS'])
        df = df.melt(id_vars=['model', 'dataset', 'precision', 'method'],
                      value_vars=['ece_delta', 'rbs_delta'],
                      var_name='ce_type',
                      value_name='CEDelta')
        df['ce_type'] = df['ce_type'].replace({
                            'ece_delta': 'ECE',
                            'rbs_delta': 'RBS'
                        })

        # Each plot is its own method
        # Separate model by color
        # Separate metric by shape
        n_hue = sorted(df['model'].unique())
        palette = dict(zip(n_hue, sns.color_palette("Dark2", n_colors=len(n_hue))))
        markers=['o', 'D', 'P', '*', 'p']

        kwargs = dict( 
                        palette=palette, markers=markers[:len(df['ce_type'].unique())], 
                        markeredgecolor='w', markersize=2.4, linewidth=2.2,
                        alpha=0.9, legend='brief'
                    )

        for i, method in enumerate(df['method'].unique()):
            sns.lineplot(
                    data=df[df['method'] == method], x="precision", y="CEDelta", 
                    hue="model", style="ce_type",  ax=ax[i],
                    **kwargs
                )
            ax[i].set_xlabel("Precision")
            ax[i].set_ylim(rc_lim)
            ax[i].set_title(f"{method}", fontsize=12)
            ax[i].get_legend().remove()

        ax[0].set_ylabel("CE Improvement")
        ax[1].set_ylabel(None)
        hdls, lbls = ax[-1].get_legend_handles_labels()

        lbls[0] = r"\textbf{Model}"
        lbls[lbls.index('ce_type')] = r"\textbf{Error metric}"

        lgd = ax[-1].legend(
            handles=hdls, labels=lbls,
            loc='upper right', bbox_to_anchor=(1.48, 0.855), #2.4, 0.5), 
            markerscale=0.6 #, ncol=5
        )
        plt.subplots_adjust(hspace=0.4, wspace=0.25, left=0.1, right=0.83, top=0.9, bottom=0.15)
        kwargs = {"bbox_extra_artists":(lgd,)}
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True, **kwargs)

    def plot_RC_deltas_by_error_deltas(
        self, settings, save_file=None, size=(8,3), save_fmt='pdf', top1_lim=[0,1.0], rc_lim=[0,0.2]
    ):
        """
        """

        settings = self._validate_dataset_settings(settings)

        df = self.results_df[
                              (self.results_df['model'].isin(settings['model']))
                            & (self.results_df['dataset'].isin(settings['dataset']))
                            & (self.results_df['precision'].isin(settings['precision']))
                            & (self.results_df['method'].isin(settings['method']))
                            ]

        # df = self._duplicate_fp_results(df, ['fq-vit', 'repq-vit'])

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=(PRECISIONS_MAP[k] for k in settings['precision'] if k in PRECISIONS_MAP),
                                         ordered=True)
        df['top1'] = df['top1'].apply(lambda x: 1.0 - (x/100.0))

        methods = df['method'].unique().tolist()
        if 'float' in methods: # it should be
            methods.remove('float')
        fig, ax = plt.subplots(1, 2, figsize=size) # there are only two RC deltas
        # fig, ax = plt.subplots(2, 2, figsize=size)

        # compute the CE delta values
        # then melt so that we can easily create plot groups by column name
        # ece_delta and rbs_delta go under 'ce_types' and their values become
        # part of another column 'ce_delta'
        df['ece_delta'] = (df['ECE'] - df['ECE_TS'])/df['ECE']
        df['rbs_delta'] = (df['RBS'] - df['RBS_TS'])/df['RBS']
        df = df.drop(columns=['top5', 'RC_probs_path', 
                              'ECE', 'ECE_TS', 'RBS', 'RBS_TS', 'NLL', 'NLL_TS'])
        df = df.melt(id_vars=['model', 'dataset', 'precision', 'method', 'top1'],
                      value_vars=['ece_delta', 'rbs_delta'],
                      var_name='ce_type',
                      value_name='CEDelta')
        df['ce_type'] = df['ce_type'].replace({
                            'ece_delta': 'ECE',
                            'rbs_delta': 'RBS'
                        })

        df_float = df[df['precision'] == 'FP32']
        result = pd.merge(df, df_float[['model','top1']], on="model", how='inner', suffixes=("", "_fp32"))
        df = result.drop_duplicates(subset=['model','precision','method','ce_type'], keep='first', ignore_index=True)
        df['Top1Delta'] = df['top1'] - df['top1_fp32'] 
        df = df.drop(df[df['precision'] == 'FP32'].index)
        df['precision'] = df['precision'].cat.remove_unused_categories()

        n_hue = sorted(df['model'].unique())
        palette = dict(zip(n_hue, sns.color_palette("Set1", n_colors=len(n_hue))))
        markers = ['o', 'p', 'D', '*', 'P']
        sizes = (20, 200)
        prec_size_dict = {}
        for sz, prec in zip(np.linspace(sizes[1],sizes[0],len(df['precision'].unique())), df['precision'].cat.categories):
            prec_size_dict[prec] = sz 

        kwargs = dict( 
                        sizes=prec_size_dict, #(1,200), #(20, 200),
                        palette=palette, markers=markers[:len(df['model'].unique())], 
                        edgecolor='k', linewidth=1.0,
                        alpha=0.75, legend='brief'
                    )

        for i, ce_type in enumerate(df['ce_type'].unique()):
            # Each plot is one metric
            # Separate model by color
            sns.scatterplot(
                    data=df[df['ce_type'] == ce_type], x="Top1Delta", y="CEDelta", 
                    hue="model", style="model", size="precision",  ax=ax[i], 
                    **kwargs
                )

            ax[i].set_xlabel("Top 1 error delta from FP32")
            ax[i].set_xlim(top1_lim)
            ylabel = "CE Improvement" if i == 0 else None
            ax[i].set_ylabel(ylabel)
            # ax[i].set_ylim(rc_lim)
            ax[i].set_title(f"{ce_type}", fontsize=12)
            ax[i].get_legend().remove()

            ax[i].grid(b=True, which='major', color='grey', linestyle='-', alpha=0.1)

        # ax[0].set_xlim([0,0.03]) 
        hdls, lbls = ax[-1].get_legend_handles_labels()

        lbls_precision = lbls[lbls.index('precision')+1:] #lbls.index('model')]
        unused_precisions = list(set(lbls_precision) - set(df['precision'].unique()))
        for prec in unused_precisions:
            del hdls[lbls.index(prec)]
            del lbls[lbls.index(prec)]

        for hdl in hdls[lbls.index('precision')+1:lbls.index('model')]:
            print(hdl._label)
            hdl.set_sizes([150]) #([hdl._sizes[0]*0.3])

        # lbls[lbls.index('method')] = r"\textbf{Quant method}"
        lbls[lbls.index('precision')] = r"\textbf{Precision}"
        lbls[lbls.index('model')] = r"\textbf{Model}"

        lgd = ax[-1].legend(
            handles=hdls, labels=lbls,
            loc='upper right', bbox_to_anchor=(1.57, 1.02), #2.4, 0.5), 
            markerscale=1.2 #, ncol=5
        )
        plt.subplots_adjust(hspace=0.4, wspace=0.22, left=0.1, right=0.81, top=0.9, bottom=0.15)
        kwargs = {"bbox_extra_artists":(lgd,)}
        # plt.show()
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True, **kwargs)


    def plot_rra_curves(
        self, settings, dataset='ImageNet1k', save_file=None, size=(8,3), save_fmt='pdf', acc_lim=[0,100]
    ):
        df = self.results_df[
                      (self.results_df['model'].isin(settings['model']))
                    & (self.results_df['precision'].isin(settings['precision']))
                    & (self.results_df['method'].isin(settings['method']))
                    & (self.results_df['dataset'] == dataset)
                    ]

        df = self._duplicate_fp_results(df, settings['method'])

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=list(PRECISIONS_MAP.values()), #['FP32', 'w8a8att32', 'w8a8att8', 'w4a4att32', 'w4a4att8'],
                                         ordered=True)
        methods = df['method'].unique().tolist()
        if 'float' in methods: # it should be
            methods.remove('float')

        # fig, axs = plt.subplots(len(settings['model']), len(methods), figsize=size)
        fig, axs = plt.subplots(len(methods), len(settings['model']), figsize=size)
        set_xlbl, set_ylbl = False, False
        # for i, model in enumerate(list(df['model'].unique())): # by row
        for i, method in enumerate(methods):
            if method == "float":
                continue

            # for j, method in enumerate(methods): # by col
            for j, model, in enumerate(list(df['model'].unique())):

                if len(methods) == 1:
                    ax = axs[j]
                elif len(df['model'].unique()) == 1:
                    ax = axs[i]
                else:
                    ax = axs[i,j]

                if i == 0:
                    ax.set_title(model) #method)

                if i == len(methods) - 1:
                    ax.set_xlabel("Response Rate (\%)")

                if j == 0:
                    ax.set_ylabel("Accuracy (\%)") #model)

                if j != 0:
                    ax.set_yticklabels([])

                palette = sns.light_palette("midnightblue", n_colors=len(settings['precision']))
                # palette = sns.dark_palette("#f8481c", n_colors=len(settings['precision']))

                # Current df slice
                for k, precisionk in enumerate(settings['precision']):
                    precision = PRECISIONS_MAP[precisionk]
                    df_slice = df[
                                    (df['method'] == method) 
                                    & (df['model'] == model)
                                    & (df['precision'] == precision)
                                ].reset_index(drop=True)

                    print(f"{model}, {method}, {precision}")
                    logits_test, logits_test_RC, labels = unpickle_probs_RC(df_slice.at[0, 'RC_probs_path'])

                    aurra_vals = 100*get_imagenet_a_results(logits_test_RC, labels, curve=True)

                    # x-axis values are by count; map each index to the actual percentage
                    response_rates = np.linspace(0, 100, len(labels))

                    ax.plot(response_rates, aurra_vals, label=precision, 
                        color=palette[k], linewidth=0.8)
                    ax.set_ylim(acc_lim)

        # Enforce order of precision in legend
        last_ax = axs[-1,-1] if len(axs.shape) > 1 else axs[-1]
        hdls, lbls = last_ax.get_legend_handles_labels()
        lgd_idx_order = [list(df['precision'].cat.categories).index(lbl) for lbl in lbls]
        
        lgd_idx_order_sort = [lgd_idx_order.index(i) for i in sorted(lgd_idx_order)]
        hdls_ord = [hdls[i] for i in lgd_idx_order_sort]
        lbls_ord = [lbls[i] for i in lgd_idx_order_sort]

        lgd = fig.legend(handles=hdls_ord, labels=lbls_ord, loc='right')

        plt.subplots_adjust(wspace=0.07)
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True) #, **kwargs)


    def plot_entropy_histograms(
        self, settings, col_setting, target_id=166, save_file=None, size=(8,3), save_fmt='pdf',
        y_lim_known=[0,0.14], y_lim_unknown=[0,0.14]
    ):
        """This function plots predictive entropy histograms for seen vs unseen classes.

        This function plots the histogram of predictive entropy on test examples from known classes 
        and unknown classes as we vary precision. See Lakshiminarayanan et. al, 2017 for more examples.
        Since this plot is visualizing for seen vs unseen classes, we should only be using ImageNet-O 
        and the counterpart as the images are unseen.
        """
        # Top row - known classes (ImageNet1k-val-for-OOD)
        # Bottom row - unknown classes (ImageNet-O)
        # Each column represents a different qmethod; one model is represented?

        assert col_setting in ['model', 'method'], "col_setting should only be switched by 'model' or 'method'"

        df = self.results_df[
                      (self.results_df['model'].isin(settings['model']))
                    & (self.results_df['precision'].isin(settings['precision']))
                    & (self.results_df['method'].isin(settings['method']))
                    & (self.results_df['dataset'] == "ImageNet-O") # dataset is fixed
                    ]

        df = self._duplicate_fp_results(df, settings['method'])

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=list(PRECISIONS_MAP.values()), #['FP32', 'w8a8att32', 'w8a8att8', 'w4a4att32', 'w4a4att8'],
                                         ordered=True)
        
        prec_ordered = [prec for prec in df['precision'].cat.categories if prec in df['precision'].unique()]
        methods = df['method'].unique().tolist()
        if 'float' in methods: # it should be
            methods.remove('float')

        if col_setting == 'method':
            assert len(df['model'].unique()) == 1
            col_cat = methods #if col_setting == 'method' else df['model']
            title_cat = df['model'].unique().tolist()[0] #if col_setting == 'method' else methods[0]
        elif col_setting == 'model':
            assert len(methods) == 1 
            col_cat = df['model'].unique()
            title_cat = methods[0]

        fig, axs = plt.subplots(2, len(col_cat), figsize=size)


        for j, col_lbl in enumerate(col_cat): # by col 
            if col_lbl == "float":
                continue

            plot_known_df = pd.DataFrame()
            plot_unknown_df = pd.DataFrame()
            for precision in df['precision'].unique():
                df_slice = df[
                                (df[col_setting] == col_lbl) 
                            &   (df['precision'] == precision)
                            ].reset_index(drop=True)

                print(f"{col_lbl}, {precision}")

                lgs_iid, lgs_ood, lbls_iid, lbls_ood = unpickle_probs_OOD(df_slice.at[0, 'ood_probs_path'])

                # Let's just try to do this with one class?
                # The class in OOD with the most samples: id 166
                # There are still only 30 samples, so we'll try removing 20 from the iid
                # lgs_ood_target_id_idx = (lbls_ood==target_id).nonzero()
                # lgs_iid_target_id_idx = (lbls_iid==target_id).nonzero()
                lgs_for_unknown_target = lgs_ood #[lgs_ood_target_id_idx]
                lgs_for_known_target = lgs_iid #[lgs_iid_target_id_idx[0][:len(lgs_for_unknown_target)]] 
                # import pdb; pdb.set_trace()

                nlls_known, nlls_unknown = [], []
                for i in range(len(lgs_for_unknown_target)):
                    nlls_known.append(entropy(softmax(lgs_for_known_target[i]), base=2))
                    nlls_unknown.append(entropy(softmax(lgs_for_unknown_target[i]), base=2))
                
                # Add to new dataframe for plotting
                plot_known_df = pd.concat([pd.DataFrame({precision: nlls_known}), plot_known_df], axis=1).reset_index(drop=True)
                plot_unknown_df = pd.concat([pd.DataFrame({precision: nlls_unknown}), plot_unknown_df], axis=1).reset_index(drop=True)

            # kdeplot easily visualizes differences between precisions
            plot_known_df_melt = pd.melt(plot_known_df, value_vars=df['precision'].unique(), var_name='precision', value_name='predictive entropy')
            plot_unknown_df_melt = pd.melt(plot_unknown_df, value_vars=df['precision'].unique(), var_name='precision', value_name='predictive entropy')
            

            # Each plot is its own method
            # Separate model by color
            # Separate metric by shape
            palette_known = dict(zip(prec_ordered, sns.dark_palette("dodgerblue", n_colors=len(prec_ordered))))
            palette_unknown = dict(zip(prec_ordered, sns.dark_palette("violet", n_colors=len(prec_ordered))))

            # 'Unknown' and 'known' plots will have same color scheme but diff shades
            sns.kdeplot(data=plot_known_df_melt, x="predictive entropy", hue="precision", 
                        ax=axs[0,j], bw_adjust=0.8, common_norm=False, palette=palette_known)
            sns.kdeplot(data=plot_unknown_df_melt, x="predictive entropy", hue="precision", 
                        ax=axs[1,j], bw_adjust=0.8, common_norm=False, palette=palette_unknown)

            # Plot stylings
            axs[0,j].set_title(col_lbl)
            axs[0,j].set_xlabel(None)
            if j != 0:
                axs[0,j].set_ylabel(None)
                axs[1,j].set_ylabel(None)
                axs[0,j].set_yticklabels([])
                axs[1,j].set_yticklabels([])

            axs[0,j].set_ylim(y_lim_known)
            axs[1,j].set_ylim(y_lim_unknown)

            # Legend stylings
            if j == len(col_cat) - 1:
                for r in range(2):
                    hdls = axs[r,j].legend_.legendHandles
                    lbls = [t.get_text() for t in axs[r,j].legend_.texts]
                    lgd_idx_order = [list(df['precision'].cat.categories).index(lbl) for lbl in lbls]
                    
                    # hdls_ord = hdls.copy()
                    # lbls_ord = lbls.copy()  
                    lgd_idx_order_sort = [lgd_idx_order.index(i) for i in sorted(lgd_idx_order)]
                    # for idx_new, hdl, lbl in zip(lgd_idx_order_sort, hdls, lbls):
                    #     hdls_ord.remove(hdl)
                    #     lbls_ord.remove(lbl)
                    #     hdls_ord.insert(idx_new, hdl)
                    #     lbls_ord.insert(idx_new, lbl)
                    hdls_ord = [hdls[i] for i in lgd_idx_order_sort]
                    lbls_ord = [lbls[i] for i in lgd_idx_order_sort]

                    axs[r,j].legend(handles=hdls_ord, labels=lbls_ord, 
                                    bbox_to_anchor=(1.01, 0.73), loc='upper left')
            else:
                axs[0,j].get_legend().remove()
                axs[1,j].get_legend().remove()

        plt.subplots_adjust(wspace=0.05)
        plt.suptitle(title_cat)
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True)

    def plot_AUPR_comparisons(self, settings, save_file=None, size=(8,3), save_fmt='pdf'):
        # assert len(settings['method']) == 1, "only plotting for one qmethod at a time"

        df = self.results_df[
                  (self.results_df['model'].isin(settings['model']))
                & (self.results_df['precision'].isin(settings['precision']))
                & (self.results_df['method'].isin(settings['method']))
                & (self.results_df['dataset'] == "ImageNet-O") # dataset is fixed
            ]

        df = self._duplicate_fp_results(df, settings['method'])

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=list(PRECISIONS_MAP.values()), #['FP32', 'w8a8att32', 'w8a8att8', 'w4a4att32', 'w4a4att8'],
                                         ordered=True)
        df['precision'] = df['precision'].cat.remove_unused_categories()
        methods = df['method'].unique().tolist()
        if 'float' in methods: # it should be
            methods.remove('float')

        for met in methods:
            df_slice = df[df['method'] == met]

            set_xlbl, set_ylbl = False, False

            # group the models together and do x-axis by size-precision? (tiny and small)
            # okay a little bit more challenging than i thought
            # actually, sns.catplot()
            # first add the size variant as a column
            df_slice = df_slice.assign(variant=lambda dfs: dfs['model'].str.split('-').str[1])
            df_slice['variant'] = df_slice['variant'].replace({'T': 'Tiny variant', 'S': 'Small variant'})
            df_slice['model'] = df_slice['model'].str.split('-').str[0]

            fig = sns.catplot(
                data=df_slice, x="variant", y="AUPR", hue="model", col='precision', kind='bar',
                height=size[1], aspect=size[0]/size[1], palette="Set2" 
            )

            # import pdb; pdb.set_trace()
            # fig.axes.set_ylabel("AUPR", fontsize=16)
            # fig.legend(fontsize=16)
            set_first = True
            for ax in fig.axes.ravel():
                if set_first:
                    ax.set_ylabel("AUPR", fontsize=20)
                    set_first = False

                ax.set_xlabel(None)
                ax.tick_params(labelsize=16)
                ax.set_title(ax.get_title().split('=')[-1], fontsize=20)
                # add annotations
                for c in ax.containers:
                    labels = [f'{(v.get_height()):.2f}' for v in c]
                    ax.bar_label(c, labels=labels, label_type='edge', fontsize=16)

            # plt.setp(fig._legend.get_texts(), fontsize=12)
            # plt.setp(fig._legend.set_title(""))
            # import pdb; pdb.set_trace()
            fig._legend.get_title().set_fontsize(18)
            for text in fig._legend.texts:
                text.set_fontsize(16)

            self.saving_plot(fig.fig, f"{save_file}_{met}", fmt=save_fmt)


    def plot_acc_vs_confidence_thr(
        self, settings, save_file=None, size=(10,8), save_fmt='pdf', conf_lim=[0.0,0.9], acc_lim=[0,100]
    ):

        df = self.results_df[
                  (self.results_df['model'].isin(settings['model']))
                & (self.results_df['precision'].isin(settings['precision']))
                & (self.results_df['method'].isin(settings['method']))
                & (self.results_df['dataset'].isin(settings['dataset']))
            ]

        df = self._duplicate_fp_results(df, settings['method'])

        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)
        df['precision'] = pd.Categorical(df['precision'],
                                         categories=list(PRECISIONS_MAP.values()), #['FP32', 'w8a8att32', 'w8a8att8', 'w4a4att32', 'w4a4att8'],
                                         ordered=True)
        df['precision'] = df['precision'].cat.remove_unused_categories()
        methods = df['method'].unique().tolist()
        if 'float' in methods: # it should be
            methods.remove('float')
        df['top1'] = df['top1'].apply(lambda x: 1.0 - (x/100.0))


        # updates the values of full_dict by averaging with corresponding values of inc_dict
        def update_acc_vs_conf_curve(full_dict, inc_dict):
            if len(full_dict) == 0:
                full_dict = inc_dict
                return full_dict

            assert inc_dict.keys() == full_dict.keys(), "dict keys do not align"

            return {k: np.mean([vf, inc_dict[k]]) for k, vf in full_dict.items()}


        fig, axs = plt.subplots(len(methods), len(df['model'].unique()), figsize=size)
        # First you need to compute the accuracy for every confidence thr
        # e.g. at 0, that's full acc. at 10, get top probs again and basically turn the ones at 0 to incorrect
        for i, method in enumerate(methods):
            
            model_settings_ord = [MODELS_MAP[k] for k in MODELS_MAP if MODELS_MAP[k] in df['model'].unique()]
            for j, model in enumerate(model_settings_ord):

                plot_acc_vs_conf = pd.DataFrame()
                for prec in df['precision'].unique():
                    print(f"{model}, {method}, {prec}")
                    df_slice = df[ 
                                    (df['model'] == model)
                                    & (df['method'] == method)
                                    & (df['precision'] == prec)
                                ].reset_index(drop=True)

                    # merge dataset results (if you want to merge known and unknown data)
                    # i think that's what Laksminarayanan et al did for the deep ensembles paper, fig6
                    # i need to get the same labels btwn the two datasets at the very least, i think
                    # that feels more right lol. other class decisions would be skewing the results
                    # however, combining logits vs taking avg after computing curve is pretty close
                    # it's def close enough for plotting purposes imo
                    # if i choose to combine logits it's like a little extra work bc i need to reset
                    # the labels to match how the smaller dataset ids classes
                    print(f"Reading ImageNet class idx mapping...")
                    with open('./imagenet1k_idx_in_imagenet-a.txt') as f:
                        class_to_idx_map = f.read().splitlines()

                    acc_vs_conf_thr_curve_all = {}
                    for dataset in settings['dataset']:
                        df_ds_slice = df_slice[df_slice['dataset']==dataset].reset_index(drop=True)
                        logits_test, _, labels = unpickle_probs_RC(df_ds_slice.at[0, 'RC_probs_path'])
                        # logits_test_iid, logits_test, labels_iid, labels = unpickle_probs_OOD(df_ds_slice.at[0, 'ood_probs_path'])                        
                        if dataset == 'ImageNet1k':
                            imagenet1k_subset_in_imagenet_a = [(lg, label) for lg, label in zip(logits_test, labels) if str(label) in class_to_idx_map]
                            logits_test = np.array([res[0] for res in imagenet1k_subset_in_imagenet_a])
                            labels = np.array([res[1] for res in imagenet1k_subset_in_imagenet_a])

                        # logits_test = np.concatenate((logits_test_iid[:2000], logits_test_ood), axis=0)
                        # labels = np.concatenate((labels_iid[:2000], labels_ood), axis=0)
                        # for logits_test, labels in zip([logits_test_iid[:2000], logits_test_ood], [labels_iid[:2000], labels_ood]):
                        
                        # Note: I tried running for imagenet-o, and it does... well? wut lol am I tripping
                        # I suppose the AUPR kind of supports that

                        acc_vs_conf_thr_curve_all = update_acc_vs_conf_curve(
                                                        acc_vs_conf_thr_curve_all, 
                                                        get_acc_vs_conf_thr_results(logits_test, labels)
                                                    )
                        
                    # Add to new dataframe for plotting
                    df_acc_vs_conf_thr = pd.DataFrame.from_dict(
                                                    acc_vs_conf_thr_curve_all, 
                                                    orient='index',
                                                    columns=[prec]
                                                )
                    if 'confidence' in plot_acc_vs_conf:
                        df_acc_vs_conf_thr = df_acc_vs_conf_thr.reset_index(drop=True)
                    else:
                        df_acc_vs_conf_thr = df_acc_vs_conf_thr.rename_axis('confidence').reset_index()

                    plot_acc_vs_conf = pd.concat([
                                            df_acc_vs_conf_thr, 
                                            plot_acc_vs_conf
                                        ], axis=1)

                plot_acc_vs_conf_melt = pd.melt(
                                            plot_acc_vs_conf, 
                                            id_vars=['confidence'],
                                            value_vars=df['precision'].unique(),
                                            var_name=['Precision'], 
                                            value_name='Acc for conf gt thr'
                                        )
                plot_acc_vs_conf_melt['confidence'] = plot_acc_vs_conf_melt['confidence'].apply(float)
                

                if len(methods) == 1:
                    ax = axs[j]
                elif len(df['model'].unique()) == 1:
                    ax = axs[i]
                else:
                    ax = axs[i,j]

                # prec_ordered = df['precision'].cat.categories
                palette = dict(zip(
                            df['precision'].cat.categories, 
                            sns.light_palette(
                                "midnightblue", 
                                n_colors=len(df['precision'].cat.categories)
                            )
                        ))
                kwargs = dict(palette=palette, marker='o', linewidth=1.0)
                sns.lineplot(
                        data=plot_acc_vs_conf_melt, 
                        x="confidence",
                        y="Acc for conf gt thr",
                        hue="Precision", 
                        ax=ax,
                        **kwargs
                    )

                # Set up plot details
                ax.set_xlim(conf_lim)
                ax.set_ylim(acc_lim)

                if i != len(methods) - 1 and len(methods) != 1:
                    ax.set_xlabel(None)
                else:
                    ax.set_xlabel("Confidence threshold $\\tau$")

                if i == 0:
                    ax.set_title(model)

                if j > 0:
                    ax.set_ylabel(None)
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("Acc for conf $>$ $\\tau$")

                ax.get_legend().remove()

        # Enforce order of precision in legend
        last_ax = axs[-1,-1] if len(axs.shape) > 1 else axs[-1]
        hdls, lbls = last_ax.get_legend_handles_labels()
        lgd_idx_order = [list(df['precision'].cat.categories).index(lbl) for lbl in lbls]
        
        hdls_ord = hdls.copy()
        lbls_ord = lbls.copy()  
        for idx_new, hdl, lbl in zip(lgd_idx_order, hdls, lbls):
            hdls_ord.remove(hdl)
            lbls_ord.remove(lbl)
            hdls_ord.insert(idx_new, hdl)
            lbls_ord.insert(idx_new, lbl)
        lgd = fig.legend(handles=hdls_ord, labels=lbls_ord, loc='right')

        plt.subplots_adjust(wspace=0.08) #, left=0.1, right=0.81, top=0.9, bottom=0.15)
        
        self.saving_plot(fig, save_file, fmt=save_fmt, tight=True)


    def tabulate_precision_error_diffs(self, settings):
        df = self.results_df[
                            (self.results_df['method'].isin(settings['method']))
                        &   (self.results_df['precision'].isin(settings['precision']))
                        &   (self.results_df['model'].isin(settings['model']))
                        &   (self.results_df['dataset'] == 'ImageNet1k')
                    ]
        df['model'] = df['model'].replace(MODELS_MAP)
        df['method'] = df['method'].replace(METHODS_MAP)
        df['precision'] = df['precision'].replace(PRECISIONS_MAP)

        df['top1_err'] = df['top1'].apply(lambda x: 1.0 - (x/100.0))
        df.drop(columns=['top1','top5','nll','RC_probs_path','NLL','NLL_TS','AURRA','AURRA_TS','ood_probs_path','AUPR'],inplace=True)
        
        # delta btwn 8/8/8 and 6/6/6, 8/8/8 and 4/8/8, 6/6/6 and 4/8/8, 6/6/6 and 8/4/4
        # formula: np.sqrt((x2-x1)**2+(y2-y1)**2)
        # np implementation: np.linalg.norm(np.array([x2,y2]) - np.array([x1,y1]))
        precision_comparisons = {
            '8/8/8 - 6/6/6': ['w8a8att8', 'w6a6att6'],
            '8/8/8 - 4/8/8': ['w8a8att8', 'w4a8att8'],
            '4/8/8 - 6/6/6': ['w4a8att8', 'w6a6att6'],
            '6/6/6 - 8/4/4': ['w6a6att6', 'w8a4att4'],
            '6/6/6 - 4/4/4': ['w6a6att6', 'w4a4att4'],
            '8/4/4 - 4/4/4': ['w8a4att4', 'w4a4att4'],
        }

        def get_err_delta(df, model, prec, err):
            return (df[(df['model']==model) & (df['precision']==prec[0])].reset_index().at[0, err] -
                df[(df['model']==model) & (df['precision']==prec[1])].reset_index().at[0, err])

        def compute_l2_error_space(df, model, precs):
            top1_delta = get_err_delta(df, model, precs, 'top1_err')
            ece_delta = get_err_delta(df, model, precs, 'ECE')
            return np.sqrt(top1_delta**2 + ece_delta**2)

        df_comps = pd.DataFrame(index=df['model'].unique(), columns=precision_comparisons.keys())
        for model in df['model'].unique():
            for comp, precs in precision_comparisons.items():
                df_comps.loc[model, comp] = compute_l2_error_space(df, model, precs)

        df_comps.columns = pd.MultiIndex.from_product([["L2 of ECE-Top1 space"], df_comps.columns])

        df_comps.to_csv(f"./results/{settings['method'][0]}_in1k_ecevtop1_l2_all.csv")
        print(df_comps)
        print(df_comps.mean())
        print(df_comps.std())

    @staticmethod
    def _duplicate_fp_results(df, methods):
        # Duplicate the FP results to the other methods
        try:
            df_dups = []
            for method in methods:
                df_dup = df[df['precision'] == '32_32_32']
                df_dup['method'] = method
                df_dups.append(df_dup)

            df = df.drop(df.index[df['method'] == 'float'])
            df = df.append(df_dups, ignore_index=True)
        except KeyError as e:
            print(f"Could not find keys 'precision' and/or 'method'!")

        return df

    @staticmethod  
    def _validate_dataset_settings(settings, multi=False):
        if 'dataset' not in settings:
            settings['dataset'] = ['ImageNet1k']

        if not multi:
            assert len(settings['dataset']) == 1, "only plotting for one dataset"
        assert 'ImageNet-O' not in settings['dataset'], "cannot plot for ImageNet-O"

        return settings

 
    def print_results_size(self, max_rows=999):
        pd.options.display.max_rows = max_rows
        print(exp.results_df.groupby(['setting', 'ce_type']).size())
        pd.options.display.max_rows = 15


if __name__ == '__main__':

    from utils import unpickle_probs
    import argparse
    import os


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--rc_logits_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--test_logits_path',
        type=str,
        nargs='*',
        required=True
    )
    parser.add_argument(
        '--df_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        required=True,
        help='String identifier of dataset'
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default='results.csv',
        help='File to append the results into; can pre-exist'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='TS',
        help='Recalibration method;',
        choices=['TS', 'ETS', 'DIAG']
    )
    # parser.add_argument(
    #     '--start_rep',
    #     type=int,
    #     default=2500,
    #     help='Repetitions of lowest sample size; must be quadratic number;'
    # )
    parser.add_argument(
        '--ce_types',
        type=str,
        nargs="+",
        default=None,
        choices=errors.keys()
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # assert (args.test_logits_path != None and args.test_dataset.lower() != "imagenet1k"), "expecting test_logits_path"
    
    # Set default test_logits_path if testing ImageNet1k
    if args.test_dataset.lower() == "imagenet1k":
        args.test_logits_path = [args.rc_logits_path]

    exp = Experiments()
    if os.path.isfile(args.save_file):
        exp.load(args.save_file)

    t = time.time()

    # First, extract the experiment settings based on the tempscale data
    # eval_adv = args.test_dataset.lower() != "imagenet1k"
    model, _, precision, qmethod = parse_logits_path(args.rc_logits_path)

    if args.test_dataset.lower() == 'imagenet-o':
        assert len(args.test_logits_path) == 2, "testing for ImageNet-O which requires two additional logits paths, but found fewer"
        model_0, dataset_0, precision_0, qmethod_0 = parse_logits_path(args.test_logits_path[0])
        model_1, dataset_1, precision_1, qmethod_1 = parse_logits_path(args.test_logits_path[1])

        # verify test paths align with each other and RC path settings
        assert model_0 == model, f"passed wrong adv logits path, model mismatch: {model_0} {model}"
        assert precision_0 == precision, f"passed wrong adv logits path, precision mismatch: {precision_0} {precision}"
        assert qmethod_0 == qmethod, f"passed wrong adv logits path, qmethod mismatch: {qmethod_0} {qmethod}"
        
        assert model_1 == model, f"passed wrong adv logits path, model mismatch: {model_1} {model}"
        assert precision_1 == precision, f"passed wrong adv logits path, precision mismatch: {precision_1} {precision}"
        assert qmethod_1 == qmethod, f"passed wrong adv logits path, qmethod mismatch: {qmethod_1} {qmethod}"

        assert model_0 == model_1, f"passed wrong adv logits path, model mismatch: {model_0} {model_1}"
        assert precision_0 == precision_1, f"passed wrong adv logits path, precision mismatch: {precision_0} {precision_1}"
        assert qmethod_0 == qmethod_1, f"passed wrong adv logits path, qmethod mismatch: {qmethod_0} {qmethod_1}"

        # figure out which test results is the iid and ood path
        iid_logits_path = args.test_logits_path[0] if dataset_0 == "ImageNet1k-val-for-OOD" else args.test_logits_path[1]
        ood_logits_path = args.test_logits_path[1] if dataset_1 == "ImageNet-O" else args.test_logits_path[0]
        a_idx = 1 if dataset_1 == "ImageNet-O" else 0 # no set order when passing args to --test_logits_path
        # dataset = "ImageNet-O" 
    else:
        # Either -1k or -A
        # Verifying that test_logits_path settings align with RC path settings
        a_model, _, a_precision, a_qmethod = parse_logits_path(args.test_logits_path[0])
        assert a_model == model, f"passed wrong adv logits path, model mismatch: {a_model} {model}"
        assert a_precision == precision, f"passed wrong adv logits path, precision mismatch: {a_precision} {precision}"
        assert a_qmethod == qmethod, f"passed wrong adv logits path, qmethod mismatch: {a_qmethod} {qmethod}"
        
        a_idx = 0
        # dataset = a_dataset

    # rc_logits = unpickle_probs(args.rc_logits_path)
    # logits_path = args.test_logits_path[a_idx] if eval_adv else args.rc_logits_path
    # logits_type = 'logits_RC' if not(eval_adv and args.test_dataset.lower() == 'imagenet-o') else 'logits_OOD'

    # Find the row that corresponds to current logits
    # Test results will be added to the same row
    print(f"\n{80*'*'}\n{model}, {args.test_dataset}, {precision}, {qmethod}")
    full_df = pd.read_pickle(args.df_path)
    ds_df = full_df[
                (full_df['model'].str.contains(model)) & 
                (full_df["dataset"] == args.test_dataset) & 
                (full_df["precision"] == precision) & 
                (full_df["method"] == qmethod)
            ].reset_index(drop=True)
    # ds_df = ds_df.drop(columns=['nll'])

    # Set up new logits path to write the calibrated test logits
    # All experiments should be using the validation subset that matches their training distribution 
    # i.e., ImageNet-1k
    rc_logits_results_path = os.path.join('results', 'logits_RC', model, 
                                    os.path.basename(args.test_logits_path[a_idx]))
    os.makedirs(os.path.dirname(rc_logits_results_path), exist_ok=True)
    print(f"Saving logit results to {rc_logits_results_path}")

    # For ImageNet-A and -O, we re-calibrate with same training distribution but 
    # Generate test results using those respective datasets
    rc_logits = unpickle_probs(args.rc_logits_path)
    adv_results = unpickle_probs(args.test_logits_path[a_idx]) #if args.test_dataset.lower() == 'imagenet-o' else None 
    ds_df = fit_logits_scale(rc_logits, ds_df, rc_logits_results_path, adv_results=adv_results)

    # For ImageNet-O, save IID and OOD logits together for AUPR
    if args.test_dataset.lower() == 'imagenet-o':
        ood_logits_results_path = os.path.join('results', 'logits_OOD', model, 
                                    os.path.basename(args.test_logits_path[a_idx]))
        os.makedirs(os.path.dirname(ood_logits_results_path), exist_ok=True)
        print(f"Saving logit results to {ood_logits_results_path}")
        ds_df = save_aupr_logits(
                                    ds_df, ood_logits_results_path, 
                                    iid_results=unpickle_probs(args.test_logits_path[a_idx-1]), 
                                    ood_results=adv_results
                                )
    # longterm TODO: add_experiment should be bettered tailored to the quantization x uncertainty flow
    # NOTE that for imagenet-a, csv will save the aurra value, but for plotting we need to call the aurra function again
    # so that we can actually get numbers per response rate (with curve=True)
    exp.add_experiment(
        metrics_df=ds_df, model=model, method=args.method, 
        ce_types=args.ce_types, seed=args.seed, adv=(args.test_dataset.lower() == 'imagenet-o')
    )

    print('Experiment done. Runtime [s]:', time.time() - t)
    exp.save(args.save_file)
