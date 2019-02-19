from phylo_autograd.substitution import HKY
import numpy as np
from numpy.testing import assert_allclose
import pytest
import autograd

test_data = [
    (
        2.0,
        np.array([0.25, 0.25, 0.25, 0.25]),
        0.1,
        np.array([0.906563342722, 0.023790645491, 0.045855366296, 0.023790645491,
                    0.023790645491, 0.906563342722, 0.023790645491, 0.045855366296,
                    0.045855366296, 0.023790645491, 0.906563342722, 0.023790645491,
                    0.023790645491, 0.045855366296, 0.023790645491, 0.906563342722]).reshape(4, 4)
    ),(
        2.0,
        np.array([0.50, 0.20, 0.2, 0.1]),
        0.1,
        np.array([0.928287993055, 0.021032136637, 0.040163801989, 0.010516068319,
                    0.052580341593, 0.906092679369, 0.021032136637, 0.020294842401,
                    0.100409504972, 0.021032136637, 0.868042290072, 0.010516068319,
                    0.052580341593, 0.040589684802, 0.021032136637, 0.885797836968]).reshape(4, 4)
    ),(
        5.0,
        np.array([0.20, 0.30, 0.25, 0.25]),
        0.1,
        np.array([0.904026219693, 0.016708646875, 0.065341261036, 0.013923872396,
                    0.011139097917, 0.910170587813, 0.013923872396, 0.064766441875,
                    0.052273008829, 0.016708646875, 0.917094471901, 0.013923872396,
                    0.011139097917, 0.077719730250, 0.013923872396, 0.897217299437]).reshape(4, 4)
    )
]

@pytest.mark.parametrize('kappa,pi,t,expected', test_data)
def test_hky_transition_probs(kappa, pi, t, expected):
    res = HKY.transition_probs(kappa, pi, t)
    assert_allclose(res, expected)

def do_test_differentiable(func, args, argnum):
    grad_func = autograd.jacobian(func, argnum=argnum)
    grad = grad_func(*args)
    assert np.shape(grad)[2:] == np.shape(args[argnum])
    assert np.all(np.isfinite(grad))


differentiable_test_data = [x[:3] + (i,) for x in test_data for i in range(3)]
@pytest.mark.parametrize('kappa,pi,t,argnum', differentiable_test_data)
def test_hky_differentiable(kappa, pi, t, argnum):
    do_test_differentiable(HKY.transition_probs, (kappa, pi, t), argnum)
