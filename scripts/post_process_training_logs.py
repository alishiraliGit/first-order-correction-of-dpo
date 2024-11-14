import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf


def get_section_tags(file):
    all_tags = set()
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            all_tags.add(v.tag)

    return all_tags


def get_section_results(file, tags):
    data = {tag: [] for tag in tags}
    print(data.keys())
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            for tag in tags:
                if v.tag == tag:
                    data[tag].append(v.simple_value)

    return data


if __name__ == '__main__':
    # ===== Settings =====
    opt_shifts = [-1, 1]
    decay_rate = 0.5
    correction_var = 0.

    exp_name = f'offline_size100000_' \
               f'shifts{"_".join(["%g" % s for s in opt_shifts])}_' \
               f'decay{"%g" % decay_rate}_' \
               f'ceestvar4corrected{"%g" % correction_var}'

    log_dir = os.path.join('..', 'data', exp_name)

    smoothness = 100

    # ===== Load logs =====
    files = glob.glob(os.path.join(log_dir, 'events*'))
    assert len(files) >= 1, f'searching for {log_dir} following files are found: ' + '\n'.join(files)
    log_path = sorted(files)[-1]

    tags = sorted([tag for tag in get_section_tags(log_path) if 'loss' in tag])
    section_results = get_section_results(log_path, tags)

    # ===== Plot =====
    plt.figure(figsize=(5*len(tags), 4))

    for i_tag, tag in enumerate(tags):
        loss = section_results[tag]

        loss = gaussian_filter1d(loss, sigma=smoothness)

        plt.subplot(1, len(tags), i_tag + 1)

        plt.plot(range(len(loss)), loss, 'k')

        plt.ylabel(tag)
        plt.xlabel('iter')

    plt.show()
