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
from ImageColor import colormap
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
    return map(lambda v: math.log(1 + v), array)

    #return [map(lambda v: math.log(1 + v), a) for a in array]
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

    # -s 7: L2-regularized logistic regression (dual)
    # -B 1: Fit a bias term
    # -q: quiet mode
    param = liblinearutil.parameter('-s 7 -B 1 -q')

    return liblinearutil.train(prob, param)


def eval_log_regr_liblinear(predictor, examples, responses, return_probs=False):

    echo('Evaluating predictor')

    # echo('Evaluating - Total examples:' + str(len(responses)) +
    #      '\tpositive: ' + str(all_pos) +
    #      '\tnegative: ' + str(len(responses) - all_pos))

    import matplotlib.pyplot as plt
    import numpy as np

    colormap = plt.cm.gist_ncar
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)

    n_positions = 2 * FEATURES_WINDOW + 1

    mean_coef, std_coef = mean_and_std(predictor.w[: n_positions * len(marks)])

    marks_to_plot = [m for m_idx, m in enumerate(marks)
                        if any(abs(mc - mean_coef) > 2 * std_coef
                            for mc in [predictor.w[m_idx * n_positions + p]
                                       for p in xrange(n_positions)])]

    num_plots = len(marks_to_plot)

    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    # print 'mean:', mean_coef, 'std:', std_coef
    for m_idx, mark in enumerate(marks):
        if mark in marks_to_plot:
            ax.plot(range(n_positions),
                    [predictor.w[m_idx * n_positions + p]
                        for p in xrange(n_positions)],
                    'o-',
                    label=mark)
    #ax.plot(range(n_positions), [predictor.intercept_] * n_positions, label='intercept')
    ax.legend()
    plt.savefig('predictor_' + str(eval_log_regr_liblinear.cnt) + '.png')
    eval_log_regr_liblinear.cnt += 1
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
eval_log_regr_liblinear.cnt = 1


def predict_from_log_regr_liblinear(predictor, features):

    # -b 1: return probability estimates
    # -q: quiet mode
    p_label, p_acc, p_val = liblinearutil.predict([], features, predictor, '-b 1 -q')

    return [c2 for c1, c2 in p_val]


def extract_features_for_logistic_regression(features, chrom, pos, ct1_prefix, ct2_prefix):
    mark_signal = features[ct1_prefix + '_mark_signal'][chrom]
    # print chrom, pos
    # print mark_signal[pos]
    n_features = len(mark_signal[pos])

    return [mark_signal[bin_idx][feature_idx]
            for feature_idx in xrange(n_features)
                for bin_idx in xrange(pos - FEATURES_WINDOW, pos + FEATURES_WINDOW + 1)]

train_predictor = train_log_regr_liblinear
predict = predict_from_log_regr_liblinear
extract_features = extract_features_for_logistic_regression
eval_predictor = eval_log_regr_liblinear


def print_eval(tp, fp, tn, fn):
    import scipy.stats

    all_pos = tp + fn
    all_examples = tp + fp + tn + fn

    sensitivity = float(tp) / all_pos
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


class Predictor:

    def train(self, features, responses):

        echo('Constructing training set for ct1_predictors')
        stats = []
        formatted_features, formatted_responses = self.format_features_and_responses(features,
                                                                                     responses,
                                                                                     ct1_prefix='ct1',
                                                                                     ct2_prefix='ct2')

        self.ct1_predictor = train_predictor(formatted_features, formatted_responses)
        stats.append(eval_predictor(self.ct1_predictor, formatted_features, formatted_responses))

        echo('Global Evaluation')
        print_eval(*map(sum, zip(*stats)))

        echo('Constructing training set for ct2_predictors')
        stats = []
        formatted_features, formatted_responses = self.format_features_and_responses(features,
                                                                                     responses,
                                                                                     ct1_prefix='ct2',
                                                                                     ct2_prefix='ct1')

        self.ct2_predictor = train_predictor(formatted_features, formatted_responses)
        stats.append(eval_predictor(self.ct2_predictor, formatted_features, formatted_responses))

        echo('Global Evaluation')
        print_eval(*map(sum, zip(*stats)))

    def _format_features_and_responses(self, features, responses, ct1_prefix, ct2_prefix, MAX_TRAINING_EXAMPLES):

        ct1_mark_signal = features[ct1_prefix + '_mark_signal']

        responses = responses[ct1_prefix + '_target_regions']
        target_is_numeric = responses['target_is_numeric']

        if not hasattr(self, 'n_features'):
            self.n_features = len(extract_features(features,
                                                   ct1_mark_signal.keys()[0],
                                                   FEATURES_WINDOW,
                                                   ct1_prefix,
                                                   ct2_prefix))

            echo('n_features:', self.n_features)

        # determine which regions to use for training
        bin_label = dict((chrom, [None] * len(ct1_mark_signal[chrom])) for chrom in ct1_mark_signal)
        prev_idx = -2
        for chrom in ct1_mark_signal:
            for bin_idx in xrange(len(ct1_mark_signal[chrom])):

                if bin_idx < FEATURES_WINDOW or bin_idx > len(ct1_mark_signal[chrom]) - FEATURES_WINDOW - 1:
                    bin_label[chrom][bin_idx] = 'chrom_end'
                    continue

                if responses[chrom][bin_idx] is None:
                    bin_label[chrom][bin_idx] = 'skip'

                elif responses[chrom][bin_idx] == 1:

                    bin_label[chrom][bin_idx] = 'pos'

                    if bin_idx == prev_idx + 1:
                        next_bin_idx = min(bin_idx + int(TO_SKIP_FACTOR * FEATURES_WINDOW) - 1, len(ct1_mark_signal[chrom]))

                        if bin_label[chrom][next_bin_idx] != 'pos':
                            bin_label[chrom][next_bin_idx] = 'skip'

                    else:

                        for idx in xrange(max(0, bin_idx - int(TO_SKIP_FACTOR * FEATURES_WINDOW)),
                                          min(bin_idx + int(TO_SKIP_FACTOR * FEATURES_WINDOW), len(ct1_mark_signal[chrom]))):

                            if bin_label[chrom][idx] != 'pos':
                                bin_label[chrom][idx] = 'skip'

                    prev_idx = bin_idx

                else:

                    if bin_label[chrom][bin_idx] is None:
                        bin_label[chrom][bin_idx] = 'neg'

        print dict((bl, sum(b == bl for chrom in bin_label for b in bin_label[chrom]))
                   for bl in set(b for chrom in bin_label for b in bin_label[chrom]))

        # count the total number of examples and the number of positive examples for each state

        labels_to_take = ['pos', 'neg']
        n_examples = sum(bl in labels_to_take for c in bin_label for bl in bin_label[c])
        n_pos_examples = sum(bl == 'pos' for c in bin_label for bl in bin_label[c])
        n_total_bins = sum(len(ct1_mark_signal[chrom]) for chrom in ct1_mark_signal)

        echo('Total bins:' + str(n_total_bins) +
             '\tExamples:' + str(n_examples) +
             '\tPositive:%d (%.5lf)\tNegative:%d (%.5lf)' %
             (n_pos_examples, float(n_pos_examples) / n_examples,
              n_examples - n_pos_examples,
              float(n_examples - n_pos_examples) / n_examples))

        example_ids = range(n_examples)
        if n_examples > MAX_TRAINING_EXAMPLES:
            echo('Downsampling examples to ' + str(MAX_TRAINING_EXAMPLES))
            example_ids = sorted(random.sample(example_ids, MAX_TRAINING_EXAMPLES))
            n_examples = MAX_TRAINING_EXAMPLES

        # features_vector = numpy.zeros((n_examples, self.n_features), dtype=numpy.float64, order='C')
        # responses_vector = numpy.zeros((n_examples, ))

        features_vector = matrix(n_examples, self.n_features, dtype="c_double")
        responses_vector = [0] * n_examples

        ex_idx = 0
        c_idx = 0

        for chrom in sorted(ct1_mark_signal):

            for bin_idx in xrange(len(ct1_mark_signal[chrom])):

                if bin_label[chrom][bin_idx] not in labels_to_take:
                    continue

                if c_idx < len(example_ids) and ex_idx == example_ids[c_idx]:

                    feat_vec = extract_features(features, chrom, bin_idx, ct1_prefix, ct2_prefix)

                    for f_idx, f_val in enumerate(feat_vec):
                        features_vector[c_idx][f_idx] = f_val

                    responses_vector[c_idx] = responses[chrom][bin_idx]
                    c_idx += 1

                ex_idx += 1

        return features_vector, responses_vector

    def format_features_and_responses(self, features, responses, ct1_prefix, ct2_prefix):

        ct1_mark_signal = features[ct1_prefix + '_mark_signal']

        responses = responses[ct1_prefix + '_target_regions']

        if not hasattr(self, 'n_features'):
            for chrom in responses:
                for bin_idx, resp in enumerate(responses[chrom]):
                    if resp is not None:

                        self.n_features = len(extract_features(features,
                                                               chrom,
                                                               bin_idx,
                                                               ct1_prefix,
                                                               ct2_prefix))
                        break
                if hasattr(self, 'n_features'):
                    break

            echo('n_features:', self.n_features)

        n_examples = sum(r is not None for chrom in responses for r in responses[chrom])
        features_vector = matrix(n_examples, self.n_features, dtype="c_double")
        responses_vector = [0] * n_examples

        ex_idx = 0

        for chrom in sorted(ct1_mark_signal):

            for bin_idx, resp in enumerate(responses[chrom]):

                if resp is None:
                    continue

                feat_vec = extract_features(features, chrom, bin_idx, ct1_prefix, ct2_prefix)

                for f_idx, f_val in enumerate(feat_vec):
                    features_vector[ex_idx][f_idx] = f_val

                responses_vector[ex_idx] = responses[chrom][bin_idx]

                ex_idx += 1

        return features_vector, responses_vector

    def predict_chrom(self, chrom, features):

        def get_scores(predictor, ct1_prefix, ct2_prefix):
            mark_signal = features[ct1_prefix + '_mark_signal'][chrom]

            # features_vector = numpy.zeros((len(mark_signal), self.n_features), dtype=numpy.float64, order='C')
            features_vector = matrix(len(mark_signal), self.n_features, dtype="c_double")

            for bin_idx in xrange(FEATURES_WINDOW, len(mark_signal) - FEATURES_WINDOW):
                feat_vec = extract_features(features, chrom, bin_idx, ct1_prefix=ct1_prefix, ct2_prefix=ct2_prefix)

                for f_idx, f_val in enumerate(feat_vec):
                    features_vector[bin_idx][f_idx] = f_val

            for score in predict(predictor, features_vector):
                yield score

        for p1, p2 in izip(get_scores(self.ct1_predictor, 'ct1', 'ct2'),
                           get_scores(self.ct2_predictor, 'ct2', 'ct1')):

            if p1 < .1 and p2 < .1:
                yield 0
            else:
                yield math.log((.1 + max(0, p1)) / (.1 + max(0, p2)), 2)

    def test(self, features, responses):

        final_stats = []
        final_probs = []
        final_responses = []

        for predictor, ct1_prefix, ct2_prefix in [(self.ct1_predictor, 'ct1', 'ct2'),
                                                  (self.ct2_predictor, 'ct2', 'ct1')]:

            echo('Constructing examples for testing for ' + ct1_prefix)
            stats = []
            probs = []
            resp = []
            formatted_features, formatted_responses = self.format_features_and_responses(features,
                                                                                         responses,
                                                                                         ct1_prefix,
                                                                                         ct2_prefix)
            tp, fp, tn, fn, p = eval_predictor(predictor,
                                               formatted_features,
                                               formatted_responses,
                                               return_probs=True)
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


def read_target_regions_bed(bed_fname, chrom_lengths, target_is_numeric=False, MAX_EXAMPLES=200000):
    echo('Reading target regions: ' + bed_fname)

    res = dict((chrom, [None if target_is_numeric else 0] * chrom_lengths[chrom])
               for chrom in chrom_lengths)

    bed_f = gzip.open(bed_fname) if bed_fname.endswith('.gz') else open(bed_fname)

    for line in bed_f:
        buf = line.strip().split()
        chrom, start, end = buf[:3]

        if chrom not in res:
            continue

        for bin_idx in xrange(int(start) / BIN_SIZE, 1 + int(end) / BIN_SIZE):
            res[chrom][bin_idx] = float(buf[-1]) if target_is_numeric else 1
    bed_f.close()

    # for binary target regions, exclude bins that are too close to positive examples
    if not target_is_numeric:
        for chrom in res:
            for bin_idx in xrange(len(res[chrom])):

                if res[chrom][bin_idx] == 1:
                    for idx in xrange(max(0, bin_idx - int(TO_SKIP_FACTOR * FEATURES_WINDOW)),
                                      min(bin_idx + int(TO_SKIP_FACTOR * FEATURES_WINDOW), len(res[chrom]))):

                        if res[chrom][idx] != 1:
                            res[chrom][idx] = None



        total_examples = sum(r is not None for chrom in res for r in res[chrom])
    if MAX_EXAMPLES is not None and total_examples > MAX_EXAMPLES:
        echo('Downsampling to', MAX_EXAMPLES, 'examples')

        to_skip = random.sample([(chrom, bin_idx) for chrom in res
                                                            for bin_idx, value in enumerate(res[chrom])
                                                                if value is not None],
                                total_examples - MAX_EXAMPLES)

        for chrom, bin_idx in to_skip:
            res[chrom][bin_idx] = None

        # mark also both ends of the chromosome for skipping
        for chrom in res:
            for bin_idx in range(FEATURES_WINDOW) + range(chrom_lengths[chrom] - FEATURES_WINDOW - 1,
                                                          chrom_lengths[chrom]):
                res[chrom][bin_idx] = None


    print 'examples per chromosome:', dict((chrom, sum(r is not None for r in res[chrom])) for chrom in res)
    return res

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-c", "--control-file", type="string", dest="ctrl_file",
                      help="Control file", metavar="FILE")

    parser.add_option("-a", "--celltype-A-target-file", type="string", dest="ct1_target_regions_fname",
                      help="Target regions for cell type A", metavar="FILE")

    parser.add_option("-b", "--celltype-B-target-file", type="string", dest="ct2_target_regions_fname",
                      help="Target regions for cell type B", metavar="FILE")

    parser.add_option("-u", "--user-static-target-file", type="string", dest="user_static_target_regions_fname",
                      help="User provided static set of target regions for both cell types", metavar="FILE")

    parser.add_option("-s", "--provided-static-target-file", type="string", dest="provided_static_target_regions_fname",
                      help="One of the provided static sets of target regions for both cell types", metavar="FILE")

    parser.add_option("-n", "--numeric-target", action="store_true", dest="target_is_numeric",
                      help="Set this option if the target file is numeric (for example, gene expression). [%default]",
                      default=False,
                      metavar="FILE")

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
    FEATURES_WINDOW = 5
    TO_SKIP_FACTOR = 1

    CHROMOSOMES_FOR_TRAINING = ['chr1', 'chr2', 'chr3', 'chr4']
    # CHROMOSOMES_FOR_TRAINING = ['chr10']
    CHROMOSOMES_FOR_PREDICTION = 'ALL'
    # CHROMOSOMES_FOR_PREDICTION = ['chr5']
    #
    # CHROMOSOMES_FOR_PREDICTION = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
    #                               'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']
    echo('Script:', __file__)
    echo('Training on:', CHROMOSOMES_FOR_TRAINING)
    echo('Testing on:', CHROMOSOMES_FOR_PREDICTION)
    echo('FEATURES_WINDOW:', FEATURES_WINDOW)

    # read in the parameters
    control = get_params(options.ctrl_file)

    ct1_mark_signal_fname = control['ct1_mark_signal_fname']
    ct2_mark_signal_fname = control['ct2_mark_signal_fname']
    chrom_lengths = read_chrom_lengths_in_bins(options.genome,
                                               get_chromosome_ids(ct1_mark_signal_fname, ct2_mark_signal_fname))




    if options.user_static_target_regions_fname:
        ct1_target_regions_fname = options.user_static_target_regions_fname
        ct2_target_regions_fname = options.user_static_target_regions_fname
    elif options.provided_static_target_regions_fname:
        ct1_target_regions_fname = get_target_fname(options.genome, options.provided_static_target_regions_fname, True)
        ct2_target_regions_fname = get_target_fname(options.genome, options.provided_static_target_regions_fname, True)
    else:
        ct1_target_regions_fname = options.ct1_target_regions_fname
        ct2_target_regions_fname = options.ct2_target_regions_fname

    ct1_target_regions = read_target_regions_bed(ct1_target_regions_fname, chrom_lengths, options.target_is_numeric)
    ct2_target_regions = read_target_regions_bed(ct2_target_regions_fname, chrom_lengths, options.target_is_numeric)

    # read the mark signal data
    ct1_mark_signal, marks1 = read_mark_signal_for_training(ct1_mark_signal_fname, ct1_target_regions)
    ct2_mark_signal, marks2 = read_mark_signal_for_training(ct2_mark_signal_fname, ct2_target_regions)
    # quantile_norm_marks(ct1_mark_signal, ct2_mark_signal)

    if marks1 != marks2:
        print 'marks don\'t match:', marks1, marks2
        exit(1)

    marks = marks1



    output_fname = options.out_prefix

    print marks
    predictor = Predictor()
    echo('Training with: ' + str(train_predictor))
    predictor.train(features={'ct1_mark_signal': ct1_mark_signal,
                              'ct2_mark_signal': ct2_mark_signal
                              },

                    responses={'ct1_target_regions': ct1_target_regions,
                               'ct2_target_regions': ct2_target_regions,
                               'target_is_numeric': options.target_is_numeric})

    del ct1_mark_signal
    del ct2_mark_signal

    echo('+' * 60)

    ct1_mark_signal_shelve = shelve.open(ct1_mark_signal_fname)
    ct2_mark_signal_shelve = shelve.open(ct2_mark_signal_fname)

    if CHROMOSOMES_FOR_PREDICTION == 'ALL':
        CHROMOSOMES_FOR_PREDICTION = sorted([key for key in ct1_mark_signal_shelve if not key.startswith('_')])

    echo('Testing on: ' + str(CHROMOSOMES_FOR_PREDICTION))

    ct1_stats = []
    ct2_stats = []

    ct1_roc_responses = []
    ct1_roc_probs = []

    ct2_roc_responses = []
    ct2_roc_probs = []

    # Re-read the target regions for the testing phase (this will shuffle the responses again)
    ct1_target_regions = read_target_regions_bed(ct1_target_regions_fname,
                                                 chrom_lengths,
                                                 options.target_is_numeric,
                                                 1000000)
    ct2_target_regions = read_target_regions_bed(ct2_target_regions_fname,
                                                 chrom_lengths,
                                                 options.target_is_numeric,
                                                 1000000)

    with open(output_fname, 'w') as out_f:

        title = os.path.split(output_fname)[1]
        out_f.write('track type=wiggle_0 name="%s" description="%s"\n' % (title, title))

        for chrom in sorted(CHROMOSOMES_FOR_PREDICTION):
            echo('Processing ' + chrom)

            ct1_chrom_mark_signal = [transform_signal(ms) for ms in ct1_mark_signal_shelve[chrom]]
            ct2_chrom_mark_signal = [transform_signal(ms) for ms in ct2_mark_signal_shelve[chrom]]

            features = {'ct1_mark_signal': {chrom: ct1_chrom_mark_signal},
                        'ct2_mark_signal': {chrom: ct2_chrom_mark_signal}}

            # test on the first 5 chromosomes that were not used for training
            if chrom in [c for c in CHROMOSOMES_FOR_PREDICTION if c not in CHROMOSOMES_FOR_TRAINING][:5]:
                stats, probs, resp = predictor.test(features=features,
                                                    responses={'ct1_target_regions': ct1_target_regions,
                                                               'ct2_target_regions': ct2_target_regions,
                                                               'target_is_numeric': options.target_is_numeric})

                ct1_stats.extend(stats[0])
                ct1_roc_probs.extend(probs[0])
                ct1_roc_responses.extend(resp[0])

                ct2_stats.extend(stats[1])
                ct2_roc_probs.extend(probs[1])
                ct2_roc_responses.extend(resp[1])

            # continue
            echo('Storing scores for ' + chrom)
            out_f.write('fixedStep  chrom=%s  start=0  step=%d  span=%d\n' % (chrom, BIN_SIZE, BIN_SIZE))

            for score in predictor.predict_chrom(chrom, features):
                out_f.write('%.5lf\n' % score)

    ct1_mark_signal_shelve.close()
    ct2_mark_signal_shelve.close()

    echo('FINAL CT1 EVALUATION')
    print_eval(*map(sum, zip(*ct1_stats)))

    echo('FINAL CT2 EVALUATION')
    print_eval(*map(sum, zip(*ct2_stats)))

    def store_roc(resp, probs, fname):
        with open(fname, 'w') as roc_f:
            pickle.dump([resp, probs], roc_f, pickle.HIGHEST_PROTOCOL)

    store_roc(ct1_roc_responses, ct1_roc_probs, output_fname + '.ct1_roc.pickle')
    store_roc(ct2_roc_responses, ct2_roc_probs, output_fname + '.ct2_roc.pickle')

    echo('Data for ROC curves is stored in: ', output_fname + '.ct1_roc.pickle', output_fname + '.ct2_roc.pickle')

    echo('Output stored in: ' + output_fname)

