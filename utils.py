"""
    Here is the place to put utility methods that are shared by the modules.
"""
import json
import os
import re

import sys
import string
import datetime
import math

BIN_SIZE = 200

ROOT_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(ROOT_DIR, 'DATA')


def get_genome_dir(genome):
    return os.path.join(DATA_DIR, genome)


def get_target_fname(genome, target, is_binary):
    return os.path.join(os.path.join(get_genome_dir(genome),
                                     'BINARY_TARGETS' if is_binary else 'NUMERIC_TARGETS'),
                        target + '.bed.gz')


def error(msg):
    print >> sys.stderr, 'ERROR: %s' % msg
    exit(1)


_rc_trans = string.maketrans('ACGT', 'TGCA')


def reverse_compl_seq(strseq):
    """ Returns the reverse complement of a DNA sequence
    """
    return strseq.translate(_rc_trans)[::-1]


def required(name, value):
    """ This method checks whether a required command line option was suplanted indeed.
        If not, prints out an error message and terminates the process.
    """
    if value is None:
        error("%s is required!" % name)


global_stime = datetime.datetime.now()


def elapsed(message = None):
    """ Measures how much time has elapsed since the last call of this method and the beginning of the execution.
        If 'message' is given, the message is printed out with the times.
    """
    print "[Last: " + str(datetime.datetime.now() - elapsed.stime) + ', Elapsed time: '+str(datetime.datetime.now() - global_stime)+ "] %s" % message if message is not None else ""
    elapsed.stime = datetime.datetime.now()
elapsed.stime = datetime.datetime.now()


def open_log(fname):
    """ Opens a log file
    """
    open_log.logfile = open(fname, 'w', 1)


def logm(message):
    """ Logs a message with a time stamp to the log file opened by open_log
    """
    print "[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message)
    open_log.logfile.write("[ %s ] %s\n" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))


def echo(*message):
    print >>sys.stderr, "[%s] %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join(map(str, message)))


def close_log():
    """ Closes the log file opened by open_log
    """
    open_log.logfile.close()


PLOT_DIR = 'plots'


def plot_d(fname):
    return os.path.join(PLOT_DIR, fname)


def listdir(path, filter = None):
    return (os.path.join(path, fname) for fname in os.listdir(path) if filter is None or re.search(filter, fname))


def matrix(n, m, default=0, dtype=None):
    if dtype is None:
        return [[default for _ in xrange(m)] for _ in xrange(n)]
    elif dtype == 'c_double':
        import ctypes
        arr = ctypes.c_double * m
        return [arr() for _ in xrange(n)]
    else:
        raise "Unknown dtype: " + dtype


def mean(array):
    return sum(array) / float(len(array))


def std(array):
    m = mean(array)
    return math.sqrt(sum((x - m) ** 2 for x in array) / float(len(array)))


def mean_and_std(array):
    m = mean(array)
    return m, math.sqrt(sum((x - m) ** 2 for x in array) / float(len(array)))


def get_params(ctrl_fname):

    with open(ctrl_fname) as control_f:
        control = json.load(control_f)

    for k in control:
        if k.endswith('fname'):
            control[k] = os.path.join(control['root'], control[k])

    return control


def read_chrom_lengths_in_bins(genome, chrom_ids=None):

    if os.path.exists(genome):
        genome_fname = genome
    else:
        genome_fname = os.path.join(os.path.join(get_genome_dir(genome), 'CHROMOSOMES'), genome + '.txt')

    chrom_lengths = {}
    with open(genome_fname) as in_f:
        for line in in_f:
            chrom, l = line.strip().split()

            if chrom_ids is not None and chrom not in chrom_ids:
                continue

            chrom_lengths[chrom] = 1 + int(l) / BIN_SIZE

    return chrom_lengths


def state_key(state):
    try:
        return int(state[1:])
    except:
        pass
    try:
        return int(state.split('_')[0])
    except:
        return state
