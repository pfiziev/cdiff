import cPickle as pickle
import os
import shelve
import sys
import gzip
from utils import *


def put_mark_signal_in_shelve(celltype, wig_files, out_fname):

    marks_signal = {}
    marks = []
    bin_size = None

    for m_fname in sorted(wig_files):

        ct, mark = os.path.split(m_fname)[1].split('.')[:2]

        marks.append(mark)

        chrom = None
        bin_idx = None

        echo('Reading signal for ' + mark + ': ' + os.path.join(m_fname))

        with (gzip.open(m_fname) if m_fname.endswith('.gz') else open(m_fname)) as mark_f:

            for line in mark_f:

                if line.startswith('track'):
                    continue

                elif line.startswith('fixedStep'):
                    m = re.search(r'span=(\d+)', line)
                    bin_size = int(m.group(1))

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
    outf['_bin_size'] = bin_size
    outf['_celltype'] = celltype

    outf.close()

    echo('Output stored in ' + out_fname)


def put_mark_signal_in_shelve_by_chromosome(wig_files, out_fname):

    marks_signal = None
    marks = []
    bin_size = None
    chrom = None
    mark_files = []
    celltype = None

    for m_fname in sorted(wig_files):

        celltype, mark = os.path.split(m_fname)[1].split('.')[:2]
        mark_files.append(gzip.open(m_fname) if m_fname.endswith('.gz') else open(m_fname))
        marks.append(mark)

    echo('marks: ', marks)

    outf = shelve.open(out_fname, protocol=pickle.HIGHEST_PROTOCOL)

    for lines in izip(*mark_files):

        if any(line.startswith('track') for line in lines):
            if not all(line.startswith('track') for line in lines):
                print 'Error:\n', lines
                exit(1)
            else:
                continue

        elif any(line.startswith('fixedStep') for line in lines):

            if marks_signal is not None:
                echo('Copying to shelve:', chrom)
                outf[chrom] = marks_signal

            if not all(line.startswith('fixedStep') for line in lines):
                print 'ERROR:\n', '\n'.join('%s: %s' % (m, l) for m, l in zip(marks, lines))
                exit(1)

            line = lines[0]

            m = re.search(r'span=(\d+)', line)
            bin_size = int(m.group(1))

            chrom = line.split()[1].split('=')[1]

            if not all(('chrom=' + chrom) in line for line in lines):
                print 'ERROR: Chromosomes do not match:\n', '\n'.join('%s: %s' % (m, l) for m, l in zip(marks, lines))

            marks_signal = []

        else:

            marks_signal.append([float(line) for line in lines])

    echo('Copying to shelve:', chrom)
    outf[chrom] = marks_signal

    outf['_marks'] = marks
    outf['_bin_size'] = bin_size
    outf['_celltype'] = celltype

    outf.close()
    map(lambda f: f.close, mark_files)

    echo('Output stored in ' + out_fname)

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print 'usage: %s wig-files output.shelve' % __file__
        exit(1)

    wig_files = sys.argv[1:-1]
    out_fname = sys.argv[-1]

    echo('N wig files:', len(wig_files))
    echo('Output:', out_fname)

    if os.path.exists(out_fname):
        error(out_fname + ' exists. Delete it if you really want to overwrite it!')

    put_mark_signal_in_shelve_by_chromosome(wig_files, out_fname)
