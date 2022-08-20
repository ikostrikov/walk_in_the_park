"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_file() to start logging to a 
tab-separated-values file (some_file_name.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""

import atexit
import errno
import json
import os
import os.path as osp
import shutil
import subprocess
import time

# import sac_dev.util.mpi_util as mpi_util


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum':
                o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {'$function': o.__module__ + "." + o.__name__}
        return json.JSONEncoder.default(self, o)


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger:
    class Entry:
        def __init__(self, val, quiet=False):
            self.val = val
            self.quiet = quiet
            return

    def print(str):
        if (Logger.is_root()):
            print(str)
        return

    def is_root():
        return True

    def __init__(self):
        self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self._dump_str_template = ""
        self._max_key_len = 0
        return

    def reset(self):
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        if self.output_file is not None:
            self.output_file = open(output_path, 'w')
        return

    def configure_output_file(self, filename=None, variant=None):
        """
        Set output directory to d, or to /tmp/somerandomnumber if d is None
        """
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        output_path = filename or "output/log_%i.txt" % int(time.time())

        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir) and Logger.is_root():
            os.makedirs(out_dir)

        if (Logger.is_root()):
            self.output_file = open(output_path, 'w')
            assert osp.exists(output_path)
            atexit.register(self.output_file.close)

            Logger.print("Logging data to " + self.output_file.name)

            if variant is not None:
                variant_log_path = osp.join(out_dir, "variant.json")
                mkdir_p(out_dir)
                with open(variant_log_path, "w") as f:
                    json.dump(variant,
                              f,
                              indent=2,
                              sort_keys=True,
                              cls=MyEncoder)
        return

    def log_tabular(self, key, val, quiet=False):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
        if self.first_row and key not in self.log_headers:
            self.log_headers.append(key)
            self._max_key_len = max(self._max_key_len, len(key))
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        self.log_current_row[key] = Logger.Entry(val, quiet)
        return

    def get_num_keys(self):
        return len(self.log_headers)

    def print_tabular(self):
        """
        Print all of the diagnostics from the current iteration
        """

        key_spacing = self._max_key_len
        format_str = "| %" + str(key_spacing) + "s | %15s |"

        if (Logger.is_root()):
            vals = []
            Logger.print("-" * (22 + key_spacing))
            for key in self.log_headers:
                entry = self.log_current_row.get(key, "")
                if not (entry.quiet):
                    val = entry.val

                    if isinstance(val, float):
                        valstr = "%8.3g" % val
                    elif isinstance(val, int):
                        valstr = str(val)
                    else:
                        valstr = val

                    Logger.print(format_str % (key, valstr))
                    vals.append(val)
            Logger.print("-" * (22 + key_spacing))
        return

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration
        """
        if (Logger.is_root()):
            if (self.first_row):
                self._dump_str_template = self._build_str_template()

            vals = []
            for key in self.log_headers:
                entry = self.log_current_row.get(key, "")
                val = entry.val
                vals.append(val)

            if self.output_file is not None:
                if self.first_row:
                    header_str = self._dump_str_template.format(
                        *self.log_headers)
                    self.output_file.write(header_str + "\r\n")

                val_str = self._dump_str_template.format(*map(str, vals))
                self.output_file.write(val_str + "\r\n")
                self.output_file.flush()

        self.log_current_row.clear()
        self.first_row = False
        return

    def has_key(self, key):
        return key in self.log_headers

    def get_current_val(self, key):
        val = None
        if (key in self.log_current_row.keys()):
            entry = self.log_current_row[key]
            val = entry.val
        return val

    def _build_str_template(self):
        num_keys = self.get_num_keys()
        template = "{:<25}" * num_keys
        return template
