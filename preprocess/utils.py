
import io

import biosppy
import numpy as np
import PIL
import pyhrv
from matplotlib import pyplot as plt
from scipy import io as sio


def load_mat(fp):
    sample = sio.loadmat(fp)['val']
    return sample

def load_header_file(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    header = [line.strip('\n') for line in header]
    return header

def signal_to_nnis(sample, sampling_rate):
    nnis = []
    for i in range(sample.shape[0]):
        try:
            _, rpeaks = biosppy.signals.ecg.ecg(sample[i, :], sampling_rate=sampling_rate, show=False)[1:3]
            nni = pyhrv.tools.nn_intervals(rpeaks)
            nnis.append(nni)
        except ValueError:
            pass
    return nnis

def poincare(nnis=None,
             rpeaks=None, marker='o',
             figsize=None, fp=None):
    # Clear the last figure
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    if figsize is None:
        figsize = (6, 6)
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    ax = fig.add_subplot(111)
    ax.set_xlim([180, 700])
    ax.set_ylim([180, 700])

    # Check input values
    for nni in nnis:
        nn = pyhrv.utils.check_input(nni, rpeaks)

        # Prepare Poincaré data
        x1 = np.asarray(nn[:-1])
        x2 = np.asarray(nn[1:])

        # ax.plot(x1, x2, markersize=2)
        ax.plot(x1, x2, '%s' % marker, markersize=5, alpha=0.5, zorder=1, color='black')
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    
    if fp is not None: 
        plt.savefig(fp, bbox_inches='tight')
        output = None
    else:
        buf = io.BytesIO()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(buf, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        output = PIL.Image.open(buf).convert('RGB')
    
    return output