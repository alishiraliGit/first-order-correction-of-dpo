import tensorflow as tf
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, log_dir, logging_freq=1):
        self.log_dir = log_dir
        self.logging_freq = logging_freq

        self.summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

        print(f'logging outputs to {self.log_dir}')

    def log(self, key: str, val, step: int):
        if step % self.logging_freq == 0:
            self.summ_writer.add_scalar(key, val, step)
            return True
        else:
            return False

    def flush(self):
        self.summ_writer.flush()


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
