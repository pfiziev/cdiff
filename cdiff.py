"""

"""
import gzip
from itertools import izip, chain
import math
from optparse import OptionParser
import os
import pprint
import random
import shelve
import sys
import gc
from utils import *
import cPickle as pickle



__author__ = 'Fiziev'


def read_mark_signal_for_training(shelve_fname, target_regions):
    echo('Reading ' + shelve_fname)

    mark_signal = {}

    shelve_f = shelve.open(shelve_fname)

    marks = shelve_f['_marks']
    for chrom in shelve_f:

        if chrom not in target_regions:
            continue

        chrom_data = shelve_f[chrom]

        # determine which bins to keep depending on the size of FEATURES_WINDOW
        to_keep = [False] * len(chrom_data)
        for bin_idx, t_reg in enumerate(target_regions[chrom]):

            if t_reg is None:
                continue

            for idx in xrange(max(0, bin_idx - FEATURES_WINDOW),
                              min(bin_idx + FEATURES_WINDOW + 1, len(chrom_data))):
                to_keep[idx] = True

        mark_signal[chrom] = [transform_signal(vals) if to_keep[bin_idx] else None
                              for bin_idx, vals in enumerate(chrom_data)]

    shelve_f.close()

    return mark_signal, marks


def transform_signal(array):
    return map(lambda v: math.log(1 + v, 2), array)

    # return [a for a in array]


def get_chromosome_ids(ct1_mark_signal_fname, ct2_mark_signal_fname):
    s1 = shelve.open(ct1_mark_signal_fname)
    s2 = shelve.open(ct2_mark_signal_fname)

    chrom_ids = sorted(set(c for c in s1 if not c.startswith('_')) &
                       set(c for c in s2 if not c.startswith('_')))

    s1.close()
    s2.close()

    return chrom_ids


def train_log_regr_scikit(features, responses):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import numpy

    n_features = len(features[0])
    # print 'n_features:', n_features

    predictor = LogisticRegression()
    # predictor = LogisticRegression(class_weight='auto')
    # predictor = RandomForestClassifier(min_samples_leaf=1000, n_jobs=4)

    # echo('Training the logistic regression..')

    # echo('Training - Total examples:' + str(len(responses)) +
    #      '\tpositive: ' + str(sum(responses)) +
    #      '\tnegative: ' + str(len(responses) - sum(responses)) +
    #      '\tn_features:' + str(n_features))

    if type(features) == numpy.ndarray:
        examples = features
    else:
        examples = numpy.array(features, dtype=numpy.float64)

    predictor.fit(examples, responses)

    return predictor


def eval_log_regr_scikit(predictor, examples, responses, return_probs=False):

    # echo('Evaluating predictor')

    # echo('Evaluating - Total examples:' + str(len(responses)) +
    #      '\tpositive: ' + str(all_pos) +
    #      '\tnegative: ' + str(len(responses) - all_pos))

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import numpy as np

    colormap = plt.cm.gist_ncar
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)

    n_positions = 2 * FEATURES_WINDOW + 1
    mean_coef, std_coef = mean_and_std(predictor.coef_[0])

    marks_to_plot = [m for m_idx, m in enumerate(marks)
                        if any(abs(mc - mean_coef) > 2 * std_coef
                            for mc in [predictor.coef_[0][m_idx * n_positions + p]
                                       for p in xrange(n_positions)])]

    num_plots = len(marks_to_plot)

    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    # print 'mean:', mean_coef, 'std:', std_coef
    for m_idx, mark in enumerate(marks):
        if mark in marks_to_plot:
            ax.plot(range(n_positions),
                [predictor.coef_[0][m_idx * n_positions + p]
                    for p in xrange(n_positions)],
                'o-',
                label=mark)
    #ax.plot(range(n_positions), [predictor.intercept_] * n_positions, label='intercept')
    ax.legend()
    plt.savefig('predictor_' + str(eval_log_regr_scikit.cnt) + '.png')
    eval_log_regr_scikit.cnt += 1
    # print 'intercept: ', predictor.intercept_
    # for mark, weight in sorted(zip(marks,
    #                                [mean_and_std([predictor.coef_[0][m_idx * (2 * FEATURES_WINDOW + 1) + p]
    #                                        for p in xrange(2 * FEATURES_WINDOW + 1)])
    #                                            for m_idx, _ in enumerate(marks)]),
    #                            key=lambda (m, w): w):
    #     print mark, weight

    # for m_idx, mark in enumerate(marks):
    #     print '\t'.join([mark, '%.5lf' % sum(predictor.coef_[0][m_idx * (2 * FEATURES_WINDOW + 1) + p]
    #                     for p in xrange(2 * FEATURES_WINDOW + 1))])

        # [mark_signal[bin_idx][feature_idx]
        #         for bin_idx in xrange(pos - FEATURES_WINDOW, pos + FEATURES_WINDOW + 1)
        #         for feature_idx in xrange(n_features)]


    probs = predictor.predict_proba(examples)[:, 1]
    predictions = [int(p >= 0.5) for p in probs]

    tp = sum(p == 1 and r == 1 for p, r in izip(predictions, responses))
    fp = sum(p == 1 and r == 0 for p, r in izip(predictions, responses))
    fn = sum(p == 0 and r == 1 for p, r in izip(predictions, responses))
    tn = sum(p == 0 and r == 0 for p, r in izip(predictions, responses))

    print_eval(tp, fp, tn, fn)

    if return_probs:
        return tp, fp, tn, fn, probs
    else:
        return tp, fp, tn, fn
eval_log_regr_scikit.cnt = 1


def predict_from_log_regr_scikit(predictor, features):
    # return predictor.predict_proba(features)[0][1]
    return predictor.predict_proba(features)[:, 1]


#### LIBLINEAR
sys.path = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'third_party',
                         'liblinear-1.94',
                         'python')] + sys.path

import liblinearutil


def train_log_regr_liblinear(features, responses):
    echo('Training with Liblinear')
    prob = liblinearutil.problem(responses, features)

    # -s 0: L2-regularized logistic regression (primal)
    # -B 1: Fit a bias term
    # -q: quiet mode
    param = liblinearutil.parameter('-s 0 -B 1 -q')

    return liblinearutil.train(prob, param)


def eval_log_regr_liblinear(predictor, examples, responses, return_probs=False, celltype_idx=-1):

    echo('Evaluating predictor')
    weight_sign = -1 if int(predictor.label[1]) == 1 else 1

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    import numpy as np

    colormap = plt.cm.gist_ncar

    n_positions = 2 * FEATURES_WINDOW + 1

    # mean_coef, std_coef = mean_and_std(predictor.w[: n_positions * len(marks)])

    # marks_to_plot = sorted([m for m_idx, m in enumerate(marks)
    #                     if any(abs(mc - mean_coef) >= 0 # std_coef
    #                         for mc in [predictor.w[m_idx * n_positions + p]
    #                                    for p in xrange(n_positions)])])

    marks_to_plot = sorted(marks,
                           key=lambda m: max([abs(predictor.w[marks.index(m) * n_positions + p])
                                                                for p in xrange(n_positions)]),
                           reverse=True)

    MAX_MARKS_PER_PLOT = 13

    if celltype_idx > 0:
        for chunk_idx in xrange(1 + len(marks) / MAX_MARKS_PER_PLOT):
            fig, ax = plt.subplots()
            fig.set_size_inches(20, 20)

            c_marks = marks_to_plot[chunk_idx * MAX_MARKS_PER_PLOT: (chunk_idx + 1) * MAX_MARKS_PER_PLOT]

            plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, MAX_MARKS_PER_PLOT)])

            for mark in c_marks:
                ax.plot(range(n_positions),
                        [weight_sign * predictor.w[marks.index(mark) * n_positions + p]
                            for p in xrange(n_positions)],
                        'o-',
                        label=mark,
                        linewidth=2.0)
            ax.plot(range(n_positions),
                    [weight_sign * predictor.w[-1] for _ in xrange(n_positions)],
                    'o-',
                    label='Bias',
                    linewidth=2.0)

            ax.legend()

            plt.savefig(output_fname + '.predictor_ct' + str(celltype_idx) + '_' + str(chunk_idx) +'.png')

        weight_matrix = [[weight_sign * predictor.w[marks.index(mark) * n_positions + p]
                            for p in xrange(n_positions)]
                                for mark in marks_to_plot]
        weight_matrix.append([weight_sign * predictor.w[-1] for _ in xrange(n_positions)])

        fig = plt.figure()
        fig.set_size_inches(20, 20)

        ax = fig.add_subplot(111)

        cax = ax.matshow(weight_matrix, interpolation='nearest', cmap=matplotlib.cm.seismic)
        fig.colorbar(cax)

        ax.set_xticklabels([''] + [str(p - FEATURES_WINDOW) for p in xrange(n_positions)])
        ax.set_yticklabels([''] + marks_to_plot + ['Bias'])
        plt.savefig(output_fname + '.predictor_heatmap_ct' + str(eval_log_regr_liblinear.cnt) + '.png')


    # print 'intercept: ', predictor.intercept_
    # for mark, weight in sorted(zip(marks,
    #                                [mean_and_std([predictor.coef_[0][m_idx * (2 * FEATURES_WINDOW + 1) + p]
    #                                        for p in xrange(2 * FEATURES_WINDOW + 1)])
    #                                            for m_idx, _ in enumerate(marks)]),
    #                            key=lambda (m, w): w):
    #     print mark, weight

    # for m_idx, mark in enumerate(marks):
    #     print '\t'.join([mark, '%.5lf' % sum(predictor.coef_[0][m_idx * (2 * FEATURES_WINDOW + 1) + p]
    #                     for p in xrange(2 * FEATURES_WINDOW + 1))])

        # [mark_signal[bin_idx][feature_idx]
        #         for bin_idx in xrange(pos - FEATURES_WINDOW, pos + FEATURES_WINDOW + 1)
        #         for feature_idx in xrange(n_features)]

    probs = predict_from_log_regr_liblinear(predictor, examples)
    predictions = [int(p >= 0.5) for p in probs]

    tp = sum(p == 1 and r == 1 for p, r in izip(predictions, responses))
    fp = sum(p == 1 and r == 0 for p, r in izip(predictions, responses))
    fn = sum(p == 0 and r == 1 for p, r in izip(predictions, responses))
    tn = sum(p == 0 and r == 0 for p, r in izip(predictions, responses))

    print_eval(tp, fp, tn, fn)

    if return_probs:
        return tp, fp, tn, fn, probs
    else:
        return tp, fp, tn, fn

def eval_log_regr_liblinear_bagged(predictors_array,
                                   examples,
                                   responses,
                                   return_probs=False,
                                   celltype_idx=-1,
                                   bagged_idx=-1):

    echo('Evaluating predictor')
    n_positions = 2 * FEATURES_WINDOW + 1
    weights = matrix(len(marks), n_positions, default=0)
    bias = 0

    if type(predictors_array) is not list:
        predictors_array = [predictors_array]

    for predictor in predictors_array:
        weight_sign = -1 if int(predictor.label[1]) == 1 else 1

        for mark_idx, mark in enumerate(marks):
            for pos in xrange(n_positions):
                weights[mark_idx][pos] += weight_sign * predictor.w[mark_idx * n_positions + pos] / len(predictors_array)
        bias += weight_sign * predictor.w[-1] / len(predictors_array)

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import numpy as np

    colormap = plt.cm.gist_ncar

    marks_to_plot = sorted(marks,
                           key=lambda m: max(map(abs, weights[marks.index(m)])),
                           reverse=True)

    MAX_MARKS_PER_PLOT = 13

    if celltype_idx != -1:
        for chunk_idx in xrange(1 + len(marks) / MAX_MARKS_PER_PLOT):
            fig, ax = plt.subplots()
            fig.set_size_inches(20, 20)

            c_marks = marks_to_plot[chunk_idx * MAX_MARKS_PER_PLOT: (chunk_idx + 1) * MAX_MARKS_PER_PLOT]

            plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, MAX_MARKS_PER_PLOT)])

            for mark in c_marks:
                ax.plot(range(n_positions),
                        weights[marks.index(mark)],
                        'o-',
                        label=mark,
                        linewidth=2.0)

            ax.plot(range(n_positions),
                    [bias] * n_positions,
                    'o-',
                    label='Bias',
                    linewidth=2.0)

            ax.legend()

            plt.savefig(output_fname + '.bagged_predictor_ct' + str(celltype_idx) +
                                       '_bag_' + str(bagged_idx) +
                                       '_chunk_' + str(chunk_idx) +'.png')

    probs = [mean(scores) for scores in izip(*map(lambda p: predict_from_log_regr_liblinear(p, examples),
                                                  predictors_array))]

    predictions = [int(p >= 0.5) for p in probs]

    tp = sum(p == 1 and r == 1 for p, r in izip(predictions, responses))
    fp = sum(p == 1 and r == 0 for p, r in izip(predictions, responses))
    fn = sum(p == 0 and r == 1 for p, r in izip(predictions, responses))
    tn = sum(p == 0 and r == 0 for p, r in izip(predictions, responses))

    print_eval(tp, fp, tn, fn)

    if return_probs:
        return tp, fp, tn, fn, probs
    else:
        return tp, fp, tn, fn


def predict_from_log_regr_liblinear(predictor, features):
    pos_class_idx = 1 if int(predictor.label[1]) == 1 else 0

    # -b 1: return probability estimates
    # -q: quiet mode
    p_label, p_acc, p_val = liblinearutil.predict([], features, predictor, '-b 1 -q')

    return [probs[pos_class_idx] for probs in p_val]


def extract_features_for_liblinear(features, chrom, pos):
    primary_mark_signal = features['primary_mark_signal'][chrom]

    n_marks = len(primary_mark_signal[pos])

    return [primary_mark_signal[bin_idx][mark_idx]
        for mark_idx in xrange(n_marks)
            for bin_idx in xrange(pos - FEATURES_WINDOW, pos + FEATURES_WINDOW + 1)]


def print_eval(tp, fp, tn, fn):
    import scipy.stats

    all_pos = tp + fn
    all_examples = tp + fp + tn + fn

    sensitivity = float(tp) / all_pos if all_pos > 0 else 1
    specificity = float(tn) / (all_examples - all_pos)
    fdr = float(fp) / (tp + fp) if (fp + fp) > 0 else 0
    expected_tp = float(all_pos ** 2) / all_examples
    echo('sensitivity: %.2lf\tspecificity: %.2lf\tFDR: %.2lf' % (sensitivity, specificity, fdr))
    echo('expected tp:', expected_tp,
         '\tlog2(tp/E(tp)):', math.log(float(tp + 1) / float(expected_tp + 1), 2),
         '\tp-val:', scipy.stats.binom.sf(tp - 1, tp + fp, float(all_pos) / all_examples),
         '\ttp:', tp,
         '\tfp:', fp,
         '\ttn:', tn,
         '\tfn:', fn)


def train_SVR_liblinear(features, responses):
    echo('Training with Liblinear')
    prob = liblinearutil.problem(responses, features)

    # -s 11: L2-regularized L2-loss support vector regression (primal)
    # -B 1: Fit a bias term
    # -q: quiet mode
    param = liblinearutil.parameter('-s 11 -B 1 -q')

    return liblinearutil.train(prob, param)


def eval_SVR_liblinear(predictor, examples, responses):
    echo('Evaluating predictor')
    predictions, p_acc, p_val = liblinearutil.predict([], examples, predictor, '-q')

    echo('RMSE:', rmse(responses, predictions),
         '\tR2:', R2(responses, predictions),
         '\tPearson R:', pearsonr(responses, predictions))

    return predictions


def predict_from_SVR_liblinear(predictor, features):

    # -b 1: return probability estimates
    # -q: quiet mode
    p_label, p_acc, p_val = liblinearutil.predict([], features, predictor, '-q')

    return p_label


class Predictor:

    def __init__(self, target_is_numeric, marks):

        # set True if the target feature is numeric (e.g. gene expression)
        self.target_is_numeric = target_is_numeric
        self.marks = marks
        self.randomized_mark_shufflings = [[random.random() <= 0.5 for _ in xrange(len(self.marks))]
                                               for _ in xrange(MAX_RANDOM_PREDICTORS)]

        if self.target_is_numeric:
            self._train_predictor = train_SVR_liblinear
            self._predict = predict_from_SVR_liblinear
            self._extract_features = extract_features_for_liblinear
            self._eval_predictor = eval_SVR_liblinear
            self._save_predictor = liblinearutil.save_model
            self._load_predictor = liblinearutil.load_model

        else:
            self._train_predictor = train_log_regr_liblinear
            self._predict = predict_from_log_regr_liblinear
            self._extract_features = extract_features_for_liblinear
            self._eval_predictor = eval_log_regr_liblinear_bagged
            self._save_predictor = liblinearutil.save_model
            self._load_predictor = liblinearutil.load_model

        self.ct1_predictor = []
        self.ct2_predictor = []

        echo('Training with:', str(self._train_predictor))

    def train(self, features, responses, celltype_idx, bagged_idx):

        formatted_features, formatted_responses = self.format_features_and_responses(features, responses)

        predictor = self._train_predictor(formatted_features, formatted_responses)
        self._eval_predictor(predictor,
                             formatted_features,
                             formatted_responses,
                             celltype_idx=celltype_idx,
                             bagged_idx=bagged_idx)

        # echo('Constructing', MAX_RANDOM_PREDICTORS, 'randomized predictors')
        # randomized_predictors = []
        # for rand_pred_idx in xrange(MAX_RANDOM_PREDICTORS):
        #     to_swap = self.randomized_mark_shufflings[rand_pred_idx]
        #     echo('Swap array:', to_swap)
        #     formatted_features, formatted_responses = self.format_features_and_responses(features,
        #                                                                                  responses,
        #                                                                                  to_swap=to_swap)
        #
        #     randomized_predictors.append(self._train_predictor(formatted_features, formatted_responses))
        #     #self._eval_predictor(self.ct1_randomized_predictors[-1], formatted_features, formatted_responses)

        if celltype_idx == 0:
            self.ct1_predictor.append(predictor)
            # self.ct1_randomized_predictors = randomized_predictors
        else:
            self.ct2_predictor.append(predictor)
            # self.ct2_randomized_predictors = randomized_predictors

    def format_features_and_responses(self, features, responses):

        primary_mark_signal = features['primary_mark_signal']

        if not hasattr(self, 'n_features'):
            for chrom in responses:
                for bin_idx, resp in enumerate(responses[chrom]):
                    if resp is not None:

                        self.n_features = len(self._extract_features(features,
                                                                     chrom,
                                                                     bin_idx))
                        break
                if hasattr(self, 'n_features'):
                    break

            echo('n_features:', self.n_features)

        n_examples = sum(r is not None for chrom in primary_mark_signal for r in responses[chrom])
        features_vector = matrix(n_examples, self.n_features, dtype="c_double")
        responses_vector = [0] * n_examples

        ex_idx = 0

        for chrom in sorted(primary_mark_signal):

            for bin_idx, resp in enumerate(responses[chrom]):

                if resp is None:
                    continue

                feat_vec = self._extract_features(features, chrom, bin_idx)

                for f_idx, f_val in enumerate(feat_vec):
                    features_vector[ex_idx][f_idx] = f_val

                responses_vector[ex_idx] = responses[chrom][bin_idx]

                ex_idx += 1

        return features_vector, responses_vector

    def predictions_to_score(self, p1, p2):
        PSEUDOCOUNT = .1 if self.target_is_numeric else .1

        if p1 < PSEUDOCOUNT and p2 < PSEUDOCOUNT:
            return p1, p2, 0
        else:
            # yield p2
            # the max(0, p1) is because predictions of numeric features can be negative and the math.log will fail
            return p1, p2, math.log((PSEUDOCOUNT + max(0, p1)) / (PSEUDOCOUNT + max(0, p2)), 2)

    def estimate_scores_under_the_null(self, ct1_mark_signal, ct2_mark_signal, responses):
        ct1_formatted_features, _ = self.format_features_and_responses(
                                                            {'primary_mark_signal': ct1_mark_signal}, responses)
        ct2_formatted_features, _ = self.format_features_and_responses(
                                                            {'primary_mark_signal': ct2_mark_signal}, responses)

        self.null_scores_pos = []
        self.null_scores_neg = []
        self.null_scores_zero = []
        for ct1_random_predictor, ct2_random_predictor in izip(self.ct1_randomized_predictors,
                                                               self.ct2_randomized_predictors):
            for p1, p2 in izip(self._predict(ct1_random_predictor, ct1_formatted_features),
                               self._predict(ct2_random_predictor, ct2_formatted_features)):

                score = self.predictions_to_score(p1, p2)
                if score > 0:
                    self.null_scores_pos.append(score)
                elif score < 0:
                    self.null_scores_neg.append(score)
                else:
                    self.null_scores_zero.append(score)

        self.null_scores_pos = sorted(self.null_scores_pos)
        self.null_scores_neg = sorted(self.null_scores_neg)

    def _predict_chrom(self, chrom, features):
        CHUNK_SIZE = 10000

        def get_scores(predictor, ct_prefix):

            mark_signal = features[ct_prefix + '_mark_signal'][chrom]
            extract_features_dict = {'primary_mark_signal': {chrom: mark_signal}}

            features_vector = matrix(CHUNK_SIZE, self.n_features, dtype="c_double")

            c_bin_idx = 0

            for chunk_idx in xrange(1 + len(mark_signal) / CHUNK_SIZE):

                start_bin_idx = chunk_idx * CHUNK_SIZE
                end_bin_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(mark_signal) - FEATURES_WINDOW)

                for bin_idx in xrange(max(start_bin_idx, FEATURES_WINDOW), end_bin_idx):
                    feat_vec_idx = bin_idx - start_bin_idx

                    feat_vec = self._extract_features(extract_features_dict, chrom, bin_idx)

                    for f_idx, f_val in enumerate(feat_vec):
                        features_vector[feat_vec_idx][f_idx] = f_val

                for score in self._predict(predictor, features_vector):
                    yield score

                    c_bin_idx += 1
                    if c_bin_idx == len(mark_signal):
                        break

        for p1, p2 in izip(get_scores(self.ct1_predictor, 'ct1'),
                           get_scores(self.ct2_predictor, 'ct2')):

            yield self.predictions_to_score(p1, p2)

    def predict_chrom(self, chrom, features):
        CHUNK_SIZE = 100000

        def get_scores(predictors, ct_prefix):

            mark_signal = features[ct_prefix + '_mark_signal'][chrom]
            extract_features_dict = {'primary_mark_signal': {chrom: mark_signal}}

            features_vector = matrix(CHUNK_SIZE, self.n_features, dtype="c_double")

            c_bin_idx = 0

            for chunk_idx in xrange(1 + len(mark_signal) / CHUNK_SIZE):

                start_bin_idx = chunk_idx * CHUNK_SIZE
                end_bin_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(mark_signal) - FEATURES_WINDOW)

                for bin_idx in xrange(max(start_bin_idx, FEATURES_WINDOW), end_bin_idx):
                    feat_vec_idx = bin_idx - start_bin_idx

                    feat_vec = self._extract_features(extract_features_dict, chrom, bin_idx)

                    for f_idx, f_val in enumerate(feat_vec):
                        features_vector[feat_vec_idx][f_idx] = f_val

                for scores in izip(*map(lambda predictor: self._predict(predictor, features_vector), predictors)):
                    yield mean(scores)

                    c_bin_idx += 1
                    if c_bin_idx == len(mark_signal):
                        break

        for p1, p2 in izip(get_scores(self.ct1_predictor, 'ct1'),
                           get_scores(self.ct2_predictor, 'ct2')):

            yield self.predictions_to_score(p1, p2)

    def test(self, features, responses):
        if self.target_is_numeric:
            return self._test_continuous_predictor(features, responses)
        else:
            return self._test_binary_predictor(features, responses)

    def _test_continuous_predictor(self, features, responses):
        results = []
        for predictor, ct_prefix in [(self.ct1_predictor, 'ct1'),
                                     (self.ct2_predictor, 'ct2')]:

            echo('Constructing examples for testing for ' + ct_prefix)
            formatted_features, formatted_responses = \
                self.format_features_and_responses(features={'primary_mark_signal': features[ct_prefix + '_mark_signal']},
                                                   responses=responses[ct_prefix + '_target_regions'])
            predictions = self._eval_predictor(predictor,
                                               formatted_features,
                                               formatted_responses)

            results.append([predictions, formatted_responses])

        return results

    def _test_binary_predictor(self, features, responses):
        final_stats = []
        final_probs = []
        final_responses = []

        for predictors_array, ct_prefix, celltype_idx in [(self.ct1_predictor, 'ct1', 0), (self.ct2_predictor, 'ct2', 1)]:

            echo('Constructing examples for testing for ' + ct_prefix)
            stats = []
            probs = []
            resp = []
            formatted_features, formatted_responses = \
                self.format_features_and_responses(features={'primary_mark_signal': features[ct_prefix + '_mark_signal']},
                                                   responses=responses[ct_prefix + '_target_regions'])

            tp, fp, tn, fn, p = self._eval_predictor(predictors_array,
                                                     formatted_features,
                                                     formatted_responses,
                                                     return_probs=True,
                                                     celltype_idx=celltype_idx,
                                                     bagged_idx='all')
            stats.append((tp, fp, tn, fn))
            probs.extend(p)
            resp.extend(formatted_responses)

            echo('Global Evaluation')

            print_eval(*map(sum, zip(*stats)))

            if len(resp) != len(probs):
                print 'ERROR', len(resp), len(probs)
                exit(1)

            final_stats.append(stats)
            final_probs.append(probs)
            final_responses.append(resp)

        return final_stats, final_probs, final_responses

    def save(self, fname_prefix):
        map(lambda (p_idx, p): self._save_predictor(fname_prefix + '.ct1_predictor_bag' + str(p_idx), p),
            enumerate(self.ct1_predictor))

        map(lambda (p_idx, p): self._save_predictor(fname_prefix + '.ct2_predictor_bag' + str(p_idx), p),
            enumerate(self.ct2_predictor))


    def load(self, fname_prefix):
        self.ct1_predictor = self._load_predictor(fname_prefix + '.ct1_predictor')
        self.ct2_predictor = self._load_predictor(fname_prefix + '.ct2_predictor')


def transform_numeric_target_feature(value):
    return math.log(1 + float(value), 2)


def read_target_regions_bed(bed_fname,
                            chrom_lengths,
                            target_is_numeric=False,
                            MAX_EXAMPLES=200000,
                            filter_chroms=ALL,
                            filter_regions=None):

    echo('Reading target regions: ' + bed_fname)
    # res = dict((chrom, [None if target_is_numeric else 0] * chrom_lengths[chrom])
    res = dict((chrom, [0] * chrom_lengths[chrom])
               for chrom in (chrom_lengths if filter_chroms == ALL else filter_chroms))

    bed_f = gzip.open(bed_fname) if bed_fname.endswith('.gz') else open(bed_fname)

    for line in bed_f:
        buf = line.strip().split()
        chrom, start, end = buf[:3]

        if chrom not in res:
            continue

        # skip locations that are not in the list of restricted regions, if such list is provided
        if filter_regions is not None and chrom not in filter_regions:
            continue

        for bin_idx in xrange(int(start) / BIN_SIZE, 1 + int(end) / BIN_SIZE):
            res[chrom][bin_idx] = transform_numeric_target_feature(buf[-1]) if target_is_numeric else 1
    bed_f.close()

    # for binary target regions, exclude bins that are too close to positive examples
    if not target_is_numeric:
        print 'positive examples:', str(sum(sum(r == 1 for r in res[chrom]) for chrom in res))
        print 'negative examples:', str(sum(sum(r == 0 for r in res[chrom]) for chrom in res))
        skipped = 0
        for chrom in res:
            for bin_idx in xrange(len(res[chrom])):

                if res[chrom][bin_idx] == 1:
                    for idx in xrange(max(0, bin_idx - int(TO_SKIP_FACTOR * FEATURES_WINDOW)),
                                      min(bin_idx + int(TO_SKIP_FACTOR * FEATURES_WINDOW), len(res[chrom]))):

                        if res[chrom][idx] != 1:
                            res[chrom][idx] = None
                            skipped += 1

        total_examples = sum(len(res[chrom]) for chrom in res)
        print 'Skipped examples:', skipped, 'out of', total_examples, '(%.5lf)' % (float(skipped) / total_examples)

    # if a list of regions is provided to restrict the training, set all
    # bins that do not fall inside those regions to None (exclude from training)
    if filter_regions is not None:
        for chrom in filter_regions:
            if chrom in res:
                for bin_idx, to_keep in enumerate(filter_regions[chrom]):
                    if not to_keep:
                        res[chrom][bin_idx] = None

                for bin_idx in xrange(len(filter_regions[chrom]), len(res[chrom])):
                    res[chrom][bin_idx] = None

    # mark also both ends of the chromosome for skipping
    for chrom in res:
        for bin_idx in range(FEATURES_WINDOW) + range(chrom_lengths[chrom] - FEATURES_WINDOW - 1,
                                                      chrom_lengths[chrom]):
            res[chrom][bin_idx] = None

    total_examples = sum(r is not None for chrom in res for r in res[chrom])
    if MAX_EXAMPLES is not None and total_examples > MAX_EXAMPLES:
        echo('Downsampling from', total_examples, 'to', MAX_EXAMPLES, 'examples')

        to_skip = random.sample([(chrom, bin_idx) for chrom in res
                                                            for bin_idx, value in enumerate(res[chrom])
                                                                if value is not None],
                                total_examples - MAX_EXAMPLES)

        for chrom, bin_idx in to_skip:
            res[chrom][bin_idx] = None

    print 'examples per chromosome:', dict((chrom, sum(r is not None for r in res[chrom])) for chrom in res)
    if not target_is_numeric:
        print 'positive examples:', str(sum(sum(r == 1 for r in res[chrom]) for chrom in res))
        print 'negative examples:', str(sum(sum(r == 0 for r in res[chrom]) for chrom in res))

    return res


def read_bed_intervals(restrict_training_fname):

    echo('Reading regions to restrict training:', restrict_training_fname)

    regions = {}

    with (gzip.open(restrict_training_fname) if restrict_training_fname.endswith('.gz')
          else open(restrict_training_fname)) as in_f:

        for line in in_f:
            buf = line.strip().split()
            chrom, start, end = buf[:3]

            if chrom not in regions:
                regions[chrom] = []

            regions[chrom].append([int(start) / BIN_SIZE, 1 + int(end) / BIN_SIZE])

    max_pos = dict((chrom, max([e for s, e in regions[chrom]])) for chrom in regions)
    res = dict((chrom, [False] * max_pos[chrom]) for chrom in regions)

    for chrom in regions:
        for start, end in regions[chrom]:
            for idx in xrange(start, end):
                res[chrom][idx] = True

    return res


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("-A", "--celltype-A-mark-signal", type="string", dest="ct1_mark_signal_fname",
                      help="Mark signal for cell type A", metavar="FILE")

    parser.add_option("-B", "--celltype-B-mark-signal", type="string", dest="ct2_mark_signal_fname",
                      help="Mark signal for cell type B", metavar="FILE")

    parser.add_option("-a", "--celltype-A-target-file", type="string", dest="ct1_target_regions_fname",
                      help="Target regions for cell type A", metavar="FILE")

    parser.add_option("-b", "--celltype-B-target-file", type="string", dest="ct2_target_regions_fname",
                      help="Target regions for cell type B", metavar="FILE")

    parser.add_option("-u", "--user-static-target-file", type="string", dest="user_static_target_regions_fname",
                      help="User provided static set of target regions for both cell types", metavar="FILE")

    parser.add_option("-s", "--provided-static-target-file", type="string", dest="provided_static_target_regions_fname",
                      help="One of the provided static sets of target regions for both cell types", metavar="FILE")

    parser.add_option("--test-A", type="string", dest="ct1_test_fname",
                      help="A bed file with target regions for cell type A for the testing phase", metavar="FILE")

    parser.add_option("--test-B", type="string", dest="ct2_test_fname",
                      help="A bed file with target regions for cell type B for the testing phase", metavar="FILE")

    parser.add_option("-n", "--numeric-target", action="store_true", dest="target_is_numeric",
                      help="Set this option if the target file is numeric (for example, gene expression). [%default]",
                      default=False,
                      metavar="FILE")

    parser.add_option("-r", "--restrict-training", type="string", dest="restrict_training_fname",
                      help="A bed file with regions to restrict the training phase.", metavar="FILE")

    parser.add_option("-g", "--genome", type="string", dest="genome",
                      help="Genome name (e.g. hg19, mm9) or path to a file with chromosome lengths", metavar="FILE")

    parser.add_option("-o", "--output-prefix", type="string", dest="out_prefix",
                      help="Output file name prefix.", metavar="FILE")

    parser.add_option("--bin-size", type="int", dest="bin_size", default=200,
                      help="The bin size. [%default]",
                      metavar="INT")

    (options, args) = parser.parse_args()

    # if no options were given by the user, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    BIN_SIZE = int(options.bin_size)
    FEATURES_WINDOW = 1000 / BIN_SIZE
    TO_SKIP_FACTOR = 1
    MAX_RANDOM_PREDICTORS = 0
    BAGGED_PREDICTORS = 10

    CHROMOSOMES_FOR_TRAINING = ALL
    # CHROMOSOMES_FOR_TRAINING = ['chr5', 'chr6',
    #                             'chr7', 'chr8', 'chr9', 'chr10']
    # CHROMOSOMES_FOR_TRAINING = ['chr3', 'chr4', 'chr5', 'chr6',
    #                             'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
    #                             'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18',
    #                             'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    # CHROMOSOMES_FOR_TRAINING = ['chr10']

    # CHROMOSOMES_FOR_PREDICTION = ['chr1', 'chr2', 'chr3', 'chr4']   # ALL
    CHROMOSOMES_FOR_PREDICTION = ALL
    # CHROMOSOMES_FOR_PREDICTION = ['chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
    #                               'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']

    echo('Script:', __file__)
    output_fname = options.out_prefix

    # read in the parameters

    ct1_mark_signal_fname = options.ct1_mark_signal_fname
    ct2_mark_signal_fname = options.ct2_mark_signal_fname

    chrom_lengths = read_chrom_lengths_in_bins(options.genome,
                                               get_chromosome_ids(ct1_mark_signal_fname, ct2_mark_signal_fname))

    ct1_mark_signal_shelve = shelve.open(ct1_mark_signal_fname)
    ALL_CHROMOSOME_IDS = sorted([key for key in ct1_mark_signal_shelve if not key.startswith('_')])
    marks = ct1_mark_signal_shelve['_marks']
    ct1_mark_signal_shelve.close()

    resolve_chrom_ids = lambda x: ALL_CHROMOSOME_IDS if x == ALL else x

    CHROMOSOMES_FOR_PREDICTION = resolve_chrom_ids(CHROMOSOMES_FOR_PREDICTION)
    CHROMOSOMES_FOR_TRAINING = resolve_chrom_ids(CHROMOSOMES_FOR_TRAINING)

    MAX_TRAINING_EXAMPLES = 200000
    MAX_TESTING_EXAMPLES = 200000

    echo('Training on:', CHROMOSOMES_FOR_TRAINING)
    echo('Testing on:', CHROMOSOMES_FOR_PREDICTION)
    echo('FEATURES_WINDOW:', FEATURES_WINDOW)
    echo('Marks:', marks)

    output_fname = options.out_prefix

    predictor = Predictor(options.target_is_numeric, marks)

    if options.restrict_training_fname:
        restrict_training_regions = read_bed_intervals(options.restrict_training_fname)
    else:
        restrict_training_regions = None

    if options.user_static_target_regions_fname:
        ct1_target_regions_fname = options.user_static_target_regions_fname
        ct2_target_regions_fname = options.user_static_target_regions_fname
    elif options.provided_static_target_regions_fname:
        ct1_target_regions_fname = get_target_fname(options.genome, options.provided_static_target_regions_fname, True)
        ct2_target_regions_fname = get_target_fname(options.genome, options.provided_static_target_regions_fname, True)
    else:
        ct1_target_regions_fname = options.ct1_target_regions_fname
        ct2_target_regions_fname = options.ct2_target_regions_fname

    def get_training_data(target_regions_fname,
                          ct1_mark_signal_fname):

        target_regions = read_target_regions_bed(target_regions_fname,
                                                 chrom_lengths,
                                                 options.target_is_numeric,
                                                 MAX_EXAMPLES=MAX_TRAINING_EXAMPLES,
                                                 filter_chroms=CHROMOSOMES_FOR_TRAINING,
                                                 filter_regions=restrict_training_regions)

        ct1_mark_signal, _ = read_mark_signal_for_training(ct1_mark_signal_fname, target_regions)

        return target_regions, ct1_mark_signal

    target_regions_array = []
    for celltype_idx, (target_regions_fname, primary_mark_signal_fname) in enumerate([
        (ct1_target_regions_fname, ct1_mark_signal_fname),
        (ct2_target_regions_fname, ct2_mark_signal_fname)]):

        echo('Training predictors for celltype:', celltype_idx)
        for bagged_idx in xrange(BAGGED_PREDICTORS):
            echo('Training bagged predictor:', bagged_idx)
            target_regions, primary_mark_signal = get_training_data(target_regions_fname,
                                                                    primary_mark_signal_fname)

            predictor.train(features={'primary_mark_signal': primary_mark_signal},
                            responses=target_regions,
                            celltype_idx=celltype_idx,
                            bagged_idx=bagged_idx)

            target_regions_array.append(target_regions)
            del primary_mark_signal

    echo('+' * 60)

    ct1_mark_signal_shelve = shelve.open(ct1_mark_signal_fname)
    ct2_mark_signal_shelve = shelve.open(ct2_mark_signal_fname)

    ct1_stats = []
    ct2_stats = []

    ct1_roc_responses = []
    ct1_roc_predictions = []

    ct2_roc_responses = []
    ct2_roc_predictions = []

    # this function returns a list of regions for the testing phase, that excludes all training examples
    # and their local neighbourhood
    def keep_for_testing(training_regions):
        res = dict((chrom, [True] * chrom_lengths[chrom]) for chrom in CHROMOSOMES_FOR_PREDICTION)
        for chrom in training_regions:
            if chrom in CHROMOSOMES_FOR_PREDICTION:
                for bin_idx, value in enumerate(training_regions[chrom]):
                    if value is not None:
                        for idx in xrange(max(0, bin_idx - int(TO_SKIP_FACTOR * FEATURES_WINDOW)),
                                          min(bin_idx + int(TO_SKIP_FACTOR * FEATURES_WINDOW), len(res[chrom]))):
                            res[chrom][idx] = False
        return res

    # Re-read the target regions for the testing phase (this will shuffle the responses again)
    ct1_target_regions = read_target_regions_bed(options.ct1_test_fname if options.ct1_test_fname
                                                 else ct1_target_regions_fname,
                                                 chrom_lengths,
                                                 options.target_is_numeric,
                                                 MAX_EXAMPLES=MAX_TESTING_EXAMPLES,
                                                 filter_chroms=CHROMOSOMES_FOR_PREDICTION,
                                                 filter_regions=keep_for_testing(target_regions_array[0]))
    ct2_target_regions = read_target_regions_bed(options.ct2_test_fname if options.ct2_test_fname
                                                 else ct2_target_regions_fname,
                                                 chrom_lengths,
                                                 options.target_is_numeric,
                                                 MAX_EXAMPLES=MAX_TESTING_EXAMPLES,
                                                 filter_chroms=CHROMOSOMES_FOR_PREDICTION,
                                                 filter_regions=keep_for_testing(target_regions_array[1]))

    # estimate score distribution under the null
    # echo('Estimating the score distribution under the null')
    # ct1_mark_signal, _ = read_mark_signal_for_training(ct1_mark_signal_fname, ct1_target_regions)
    # ct2_mark_signal, _ = read_mark_signal_for_training(ct2_mark_signal_fname, ct1_target_regions)
    #
    # predictor.estimate_scores_under_the_null(ct1_mark_signal, ct2_mark_signal, ct1_target_regions)
    #
    # del ct1_mark_signal, ct2_mark_signal

    real_scores = []

    ct1_probs_f = open(output_fname.replace('.wig', '') + '.ct1_probs.wig', 'w')
    ct2_probs_f = open(output_fname.replace('.wig', '') + '.ct2_probs.wig', 'w')
    rand_out_f = open(output_fname + '.rand_preds', 'w')

    with open(output_fname, 'w') as out_f:

        title = os.path.split(output_fname)[1]
        out_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title, title))
        ct1_probs_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title + ' ct1 probs', title + ' ct1 probs'))
        ct2_probs_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title + ' ct2 probs', title + ' ct2 probs'))

        for chrom in sorted(CHROMOSOMES_FOR_PREDICTION):
            echo('Processing ' + chrom)

            ct1_chrom_mark_signal = [transform_signal(ms) for ms in ct1_mark_signal_shelve[chrom]]
            ct2_chrom_mark_signal = [transform_signal(ms) for ms in ct2_mark_signal_shelve[chrom]]

            features = {'ct1_mark_signal': {chrom: ct1_chrom_mark_signal},
                        'ct2_mark_signal': {chrom: ct2_chrom_mark_signal}}

            # test on the first 5 chromosomes that were not used for training
            #if chrom in CHROMOSOMES_FOR_TRAINING:
            if chrom != 'chrM':
                test_results = predictor.test(features=features,
                                              responses={'ct1_target_regions': ct1_target_regions,
                                                         'ct2_target_regions': ct2_target_regions})

                if options.target_is_numeric:
                    [[ct1_pred, ct1_resp], [ct2_pred, ct2_resp]] = test_results

                    ct1_roc_predictions.extend(ct1_pred)
                    ct1_roc_responses.extend(ct1_resp)

                    ct2_roc_predictions.extend(ct2_pred)
                    ct2_roc_responses.extend(ct2_resp)

                else:
                    stats, probs, resp = test_results

                    ct1_stats.extend(stats[0])
                    ct1_roc_predictions.extend(probs[0])
                    ct1_roc_responses.extend(resp[0])

                    ct2_stats.extend(stats[1])
                    ct2_roc_predictions.extend(probs[1])
                    ct2_roc_responses.extend(resp[1])

            #
            echo('Storing scores for ' + chrom)
            out_f.write('fixedStep  chrom=%s  start=0  step=%d  span=%d\n' % (chrom, BIN_SIZE, BIN_SIZE))
            ct1_probs_f.write('fixedStep  chrom=%s  start=0  step=%d  span=%d\n' % (chrom, BIN_SIZE, BIN_SIZE))
            ct2_probs_f.write('fixedStep  chrom=%s  start=0  step=%d  span=%d\n' % (chrom, BIN_SIZE, BIN_SIZE))

            for p1, p2, score in predictor.predict_chrom(chrom, features):
                out_f.write('%.5lf\n' % score)
                ct1_probs_f.write('%.5lf\n' % p1)
                ct2_probs_f.write('%.5lf\n' % p2)
                real_scores.append(score)
                # rand_out_f.write('%s\t%d\t%d\t%s\n' % (chrom,
                #                                        bin_idx * BIN_SIZE,
                #                                        (bin_idx + 1) * BIN_SIZE,
                #                                        '\t'.join('%.5lf' % s for s in scores)))

    ct1_mark_signal_shelve.close()
    ct2_mark_signal_shelve.close()

    # # draw null vs real scores
    # echo('Drawing figures for random vs real scores')
    # real_pos_scores = sorted([s for s in real_scores if s > 0])
    # real_neg_scores = sorted([s for s in real_scores if s < 0])
    # import matplotlib.pyplot as plt
    # plt.figure()
    # n_bins = 100
    #
    # plt.hist(predictor.null_scores_pos, n_bins, normed=1, facecolor='blue', alpha=.5, label='random_pos')
    # plt.hist(predictor.null_scores_neg, n_bins, normed=1, facecolor='blue', alpha=.5, label='random_neg')
    # plt.hist(real_pos_scores, n_bins, normed=1, facecolor='red', alpha=.5, label='real_pos')
    # plt.hist(real_neg_scores, n_bins, normed=1, facecolor='red', alpha=.5, label='real_neg')
    # plt.legend()
    #
    # plt.savefig(output_fname + '.null_scores_dist.png')
    # print max(real_scores), min(real_scores), max(predictor.null_scores_pos), min(predictor.null_scores_neg)
    # with open(output_fname + '.scores_dist.pickle', 'w') as scores_f:
    #     pickle.dump({'real': real_scores,
    #                  'null_pos': predictor.null_scores_pos,
    #                  'null_neg': predictor.null_scores_neg}, scores_f)
    #
    # def plot_qq(scores1, scores2, title, xlabel, ylabel):
    #     import numpy as np
    #     plt.figure()
    #
    #     min_range = min(scores1 + scores2)
    #     max_range = max(scores1 + scores2)
    #     bin_width = (max_range - min_range) / float(n_bins)
    #
    #     def cum(h):
    #         h = list(h)
    #         for i in xrange(1, len(h)):
    #             h[i] += h[i - 1]
    #         return [hh * bin_width for hh in h]
    #
    #     h1, _ = np.histogram(scores1, n_bins, range=(min_range, max_range), density=True)
    #     h2, _ = np.histogram(scores2, n_bins, range=(min_range, max_range), density=True)
    #
    #     h1 = cum(h1)
    #     h2 = cum(h2)
    #
    #     plt.plot(h1, h2, linewidth=2.0)
    #     plt.plot([0, 1], [0, 1], 'r-')
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.title(title)
    #     plt.savefig(output_fname + '.' + title + 'null_scores_qq_plot.png')
    #
    # plot_qq(predictor.null_scores_pos, real_pos_scores, 'positive_scores', 'Random', 'Real')
    # plot_qq(predictor.null_scores_neg, real_neg_scores, 'negative_scores', 'Random', 'Real')

    # store ROC statistics for binary targets
    if options.target_is_numeric:
        echo('FINAL CT1 EVALUATION')
        echo('RMSE:', rmse(ct1_roc_responses, ct1_roc_predictions),
             '\tR2:', R2(ct1_roc_responses, ct1_roc_predictions),
             '\tPearson R:', pearsonr(ct1_roc_responses, ct1_roc_predictions))

        echo('FINAL CT2 EVALUATION')
        echo('RMSE:', rmse(ct2_roc_responses, ct2_roc_predictions),
             '\tR2:', R2(ct2_roc_responses, ct2_roc_predictions),
             '\tPearson R:', pearsonr(ct2_roc_responses, ct2_roc_predictions))


    else:
        echo('FINAL CT1 EVALUATION')
        print_eval(*map(sum, zip(*ct1_stats)))

        echo('FINAL CT2 EVALUATION')
        print_eval(*map(sum, zip(*ct2_stats)))

    def store_roc(resp, probs, fname):
        with open(fname, 'w') as roc_f:
            pickle.dump([resp, probs], roc_f, pickle.HIGHEST_PROTOCOL)

    store_roc(ct1_roc_responses, ct1_roc_predictions, output_fname + '.ct1_roc.pickle')
    store_roc(ct2_roc_responses, ct2_roc_predictions, output_fname + '.ct2_roc.pickle')

    echo('Data for ROC curves is stored in: ', output_fname + '.ct1_roc.pickle', output_fname + '.ct2_roc.pickle')
    predictor.save(output_fname + '.models')
    echo('Output stored in: ' + output_fname)

