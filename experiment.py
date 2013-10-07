#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import subprocess
from multiprocessing import Pool


def run_experiment(args):
    cmd = ("python hw3_runner.py --theta1 {1} --theta2 {2} --theta3 {3} "
           "-o output-{0}.txt").format(*args)
    print("Running:\n\t{0}".format(cmd))
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    with open("stdout-{0}.txt".format(args[0]), "w") as f:
        f.write("{0}\n".format(cmd))
        f.write(stdout)


if __name__ == "__main__":
    pool = Pool()
    pool.map(run_experiment, [(1, 0.05, 0.6, 0.3),
                              (2, 0.05, 0.5, 0.4),
                              (3, 0.05, 0.4, 0.5),
                              (4, 0.05, 0.3, 0.6),
                              ])
