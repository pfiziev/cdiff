import cPickle as pickle
import os
import shelve
import sys
from utils import *


def put_mark_signal_in_shelve(wig_files, out_fname):

    marks_signal = {}
    marks = []

    for m_fname in sorted(wig_files):

        ct, mark, _ = os.path.split(m_fname)[1].split('.')

        marks.append(mark)

        chrom = None
        bin_idx = None

        echo('Reading signal for ' + mark + ': ' + os.path.join(m_fname))

        with open(m_fname) as mark_f:

            for line in mark_f:

                if line.startswith('track'):
                    continue

                elif line.startswith('fixedStep'):
                    chrom = line.split()[1].split('=')[1]

                    if chrom not in marks_signal:
                        marks_signal[chrom] = []

                    bin_idx = 0

                else:

                    if bin_idx >= len(marks_signal[chrom]):
                        marks_signal[chrom].append([])

                    marks_signal[chrom][bin_idx].append(float(line))
                    bin_idx += 1

    echo('Copying the signal to the shelve')
    echo('marks: ', marks)

    outf = shelve.open(out_fname, protocol=pickle.HIGHEST_PROTOCOL)

    for chrom in marks_signal:
        outf[chrom] = marks_signal[chrom]

    outf['_marks'] = marks
    outf.close()

    echo('Output stored in ' + out_fname)

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print 'usage: %s wig-files output.shelve' % __file__
        exit(1)

    wig_files = sys.argv[1:-1]
    out_fname = sys.argv[-1]
    echo('Wig files:', ' '.join(wig_files))
    echo('Output:', out_fname)

    if os.path.exists(out_fname):
        error(out_fname + ' exists. Delete it if you really want to overwrite it!')

    put_mark_signal_in_shelve(wig_files, out_fname)
