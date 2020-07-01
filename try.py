# test MAB algo
from sampling import *

mab = MAB(4,1,0.05,0)

mab.arms

mab.should_stop()

mab.nextpull()

mab.update([1], [1])

mab.arms


# test running MAB
from single_run import *

run = single_run(4, 1, [0.085, 0.1, 0.09, 0.07], [100*(i+1) for i in range(100)])

run.run_MAB(0.05)
