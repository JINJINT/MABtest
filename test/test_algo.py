import pytest
from .uniform import uniform
from .UCB import UCB
from .TS import TS
from .uniform_decay import uniform_decay
from .UCB_decay import UCB_decay
from .TS_decay import TS_decay
from .uniform_CD import uniform_CD
from .UCB_CD import UCB_CD


def test_algo_class():
    algo = uniform(no_arms = 5)
    algo.start()
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assertTrue(algo.should_stop() in [True, False])

    algo = UCB(no_arms = 5)
    algo.start()
    assertEqual(algo.trade_off, 1.)
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assertTrue(algo.should_stop() in [True, False])

    algo = TS(no_arms = 5)
    algo.start()
    assertEqual(algo.trade_off, 1.)
    assert all(algo.prior_para_1, np.repeat(1,5))
    assert all(algo.prior_para_2, np.repeat(1,5))
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assertTrue(algo.should_stop() in [True, False])

    algo = uniform_decay(no_arms = 5)
    algo.start()
    assert algo.time_decay == 0.001
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assertTrue(algo.should_stop() in [True, False])

    algo = UCB_decay(no_arms = 5)
    algo.start()
    assert algo.time_decay == 0.001
    assertEqual(algo.trade_off, 1.)
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assertTrue(algo.should_stop() in [True, False])

    algo = TS_decay(no_arms = 5)
    algo.start()
    assert algo.time_decay == 0.001
    assertEqual(algo.trade_off, 1.)
    assert all(algo.prior_para_1, np.repeat(1,5))
    assert all(algo.prior_para_2, np.repeat(1,5))
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assertTrue(algo.should_stop() in [True, False])

    algo = uniform_CD(no_arms = 5)
    algo.start()
    assert algo.time_window == 1000
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assert all(idx in range(5) for idx in algo.next_pull(0,1))
    assertTrue(algo.should_stop() in [True, False])

    algo = UCB_CD(no_arms = 5)
    algo.start()
    assert algo.time_window == 1000
    assertEqual(algo.trade_off, 1.)
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assert all(idx in range(5) for idx in algo.next_pull(0,1))
    assertTrue(algo.should_stop() in [True, False])

    algo = TS_CD(no_arms = 5)
    algo.start()
    assert algo.time_window == 1000
    assertEqual(algo.trade_off, 1.)
    assert all(algo.prior_para_1, np.repeat(1,5))
    assert all(algo.prior_para_2, np.repeat(1,5))
    algo.next_pull()
    algo.update(0,1) 
    assertEqual(len(algo.next_pull(0,1)), algo.no_output)
    assert all(idx in range(5) for idx in algo.next_pull(0,1))
    assertTrue(algo.should_stop() in [True, False])





def test_main():


    pass






















