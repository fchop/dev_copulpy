#!/usr/bin/env python
"""Test."""
# from subprocess import CalledProcessError
# import pickle as pkl
# import subprocess
# import os


import numpy as np
# import pytest

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.shared.auxiliary import distribute_copula_spec
# from copulpy.config_copulpy import PACKAGE_DIR
np.random.seed(123)

for _ in range(10):
    x, y, is_normalized, copula_spec = generate_random_request({'version': 'nonstationary'})

    # copula_spec['marginals'] = ['exponential', 'exponential']
    # copula_spec['r'] = [-5, -5]
    # copula_spec['bounds'] = [500, 500]

    # print(is_normalized, 'out')
    copula = UtilityCopulaCls(copula_spec)
    util = copula.evaluate(x, y, is_normalized)

    alpha, beta, gamma, y_scale, version = \
        distribute_copula_spec(copula_spec, 'alpha', 'beta', 'gamma', 'y_scale', 'version')

    print('version: {}'.format(version))
    print('alpha: {0:.2f}, beta: {1:.2f}, gamma: {2:.2f}, y_scale: {3:.2f}'.format(
        alpha, beta, gamma, y_scale)
    )
    print('x: {0:.2f}, y: {1:.2f}, utility: {2:.2f}.'.format(x, y, util))

    # Don't expect monotonicity here
    print(copula.evaluate(x, y))
    print(copula.evaluate(x, y, t=0))
    print(copula.evaluate(x, y, t=1))
    print(copula.evaluate(x, y, t=3))
    print(copula.evaluate(x, y, t=6))
    print(copula.evaluate(x, y, t=12))
    print(copula.evaluate(x, y, t=24))

    print('\n')
