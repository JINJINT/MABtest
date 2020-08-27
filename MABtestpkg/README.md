# MABtest

An open-source package for online experiments using Multi-armed bandits. For questions, contact the author jinjin(at)Andrew(dot)cmu(dot)edu.

## List of algorithms

Unchanged enviroment (i.e. without time variation):

- "Stupid" algorithms: [`uniform`](uniform.py)

- "Adaptive" algorithms based on UCB, also include a trade-off parameter controlling the extend of exploration: [`UCB`](UCB.py)

- "Adaptive" algorithms based on Thompson Sampling, also include a trade-off parameter controlling the extend of exploration: [`Uniform`](TS.py)

Dynamically changing enviroment (i.e. with time variation):

1) For general unknown changes: we modify the algorithms built for `Unchanged environment` using a memory decay idea, which enforces the algorithms to gradually forget the long-arrived samples.

- "Stupid" algorithms: [`uniform-decay`](uniform-decay.py)

- "Adaptive" algorithms based on UCB, also include a trade-off parameter controlling the extend of exploration: [`UCB-decay`](UCB-decay.py)

- "Adaptive" algorithms based on Thompson Sampling, also include a trade-off parameter controlling the extend of exploration: [`TS-decay`](TS-decay.py)

2) For abrupt changes: we add to the algorithms built for `Unchanged environment` with a change detector, which enforces the algorithms to restart all over again when detect a change.

- "Stupid" algorithms: [`uniform-CD`](uniform-CD.py)

- "Adaptive" algorithms based on UCB, also include a trade-off parameter controlling the extend of exploration: [`UCB-CD`](UCB-CD.py)

- "Adaptive" algorithms based on Thompson Sampling, also include a trade-off parameter controlling the extend of exploration: [`TS`](TS-CD.py)


##  Run single experiments from terminal
Run a experiments with different configurations of the simulation setting and algorithms choice from terminal, to see detailed description of all the input argument, see [`main`](main.py).

### Quick start: 
Simulate environment with 5 treatments with independent Bernoulli observations whose mean is 0.1, 0.2, .. 0.5 respectively. Using naive uniform sampling algorithm, targeting at finding the best treatment with 95% confidence.
```console
   cd ~/MABtestpkg
   python main.py 
```
### To choose from more setting of experiments:   

#### 1. Configuration of observation type
- type 1 (default): [`Bernoulli`](https://en.wikipedia.org/wiki/Bernoulli_distribution)
```console
python main.py
```
- type 2: [`Gaussian`](https://en.wikipedia.org/wiki/https://en.wikipedia.org/wiki/Normal_distribution)
```console
python main.py --reward-type 'Gaussian'
```

#### 2. Configuration of mean of observations
##### Stationary setting: mean does not changes with time

- setting 1 (default). five Bernoulli reward with big gap in the mean: each arm has reward follows a Bernoulli distribution with mean in [0,1. 0,2, 0.3, 0.4, 0.5].

```console
   python main.py
```
- setting 2. five Bernoulli reward with small gap in the mean: each arm has reward follows a bernoulli distribution with mean in [0,11. 0,12, 0.13, 0.14, 0.15].
```console   
    python main.py --mu-type 'smallgap' --mu-list '0.11,0.12,0.13,0.14,0.15'
```
- setting 3. five bernoulli reward with random gap in the mean: each arm has reward follows a bernoulli distribution with mean generated randomly from a uniform distribution.
```console 
   python main.py --mu-type 'rand' 
```
##### Non-stationary setting: mean changes with time
- setting 1 (default). five bernoulli reward with big gap in the mean with continuous time variation: the trend follows a consine function.
   
```console
   python main.py --is-timevar 
```
- setting 2. five bernoulli reward with big gap in the mean with abrupt time variation: the trend follows a piece-wise flat function.
   
```console   
   python main.py --is-timevar --timevar-type 'Abrupt'
```
- setting 3. five bernoulli reward with small gap in the mean with abrupt time variation originated from the [`Yahoo! dataset`](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49).
```console   
   python main.py --is-timevar --timevar-type 'RealAbrupt' --record-max 4000000 --time-window 1000
```



### To choose from different sampling algorithms
#### Classical optimal solutions for minimizing revenue loss for [`MAB`](https://en.wikipedia.org/wiki/Multi-armed_bandit) problem.
- choice 1: Thompson Sampling
```console
   python main.py --samp-type 'TS'
```
- choice 2: UCB
```console   
   python main.py --samp-type 'UCB'
```
#### To enforce more exploration on sub-optimal arms to reach significance of testing faster, though at cost of higher revenue loss.
e.g. Thompson Sampling with trade-off > 1 (i.e more exploration on the sub-optimal arms, set at 1 recovers Thompson Sampling.)
```console
   python main.py --trade-off 3
```
#### To adjust for non-stationarity
- For general unknown non-stationarity, we use `memory decay` idea (when set the time var setting as 'General', the algorithms will automatically choose the memory decay idea), the decay rate can ranges from (0,1), and the default is 0.001. The higher decay rate is, the faster the algorithms forget about long-arrived samples.
```console
   python main.py --time-decay 0.01 
```
- For a particular type of non-stationarity, the piece-wise stationary, we use `detect change and restart` idea (when set the time var setting as 'Abrupt', the algorithms will automatically choose the change detect idea), where the algorithms decide whether there is a change point up to now or not based on the difference in the recently arrived samples (i.e. inside a time window from present moment), and long-arrived samples. The default detection window size is 200 (nearly optimal for the `biggap` setting in mean configuration), and we recommend 2000 for `smallgap` setting in mean configuration.
```console
   python main.py --time-window 2000 
```
### To choose from different testing problems
- Problem 1: Find the best k treatment with confidence
```console
   python main.py --no-output k --stop-type 'Best'
```
- Problem 2: Find the best k treatment over baseline with confidence
```console
   python main.py --no-output k --stop-type 'Best-Base'
```
- Problem 3: Find at least k effective treatments over baseline with [`FWER`] (https://en.wikipedia.org/wiki/Family-wise_error_rate) control
```console
   python main.py --no-output k --stop-type 'FWER-Base' --baseline 0.1
```
- Problem 4: Find at least k effective treatments over baseline with [`FDR`](https://en.wikipedia.org/wiki/False_discovery_rate) control
```console
   python main.py --no-output k --stop-type 'FDR-Base' --baseline 0.1
```
Remark: the confidence level can also be specified, and the default is 0.05 (which gives out 95% confidence level); 
```console
   python main.py --conf-level 0.1
```
Remark: the testing problems can be slightly modified for users convenience by specifying different precision level, which raise/lower the bar of being `better` by setting at positive/negative value.
```console
   python main.py --precision-level 0.1
   python main.py --precision-level -0.1
```

### Configuration of saving and plotting choice
In order to NOT save, NOT print out results, NOT plotting in the end, you can specify the following arguments
```console
   python main.py --nosave --noprint --noplot
```


## Run one bandit from python
All policies have the same interface, as described in [`uniform`](uniform.py),
in order to run them use the following approach in python:

```python
   from MABtest.uniform import *
   alg = uniform(5) # can be replaced by other algo listed above
   alg.start()  # start the bandit algo
   for t in range(100):
       if not alg.should_stop():  # if have not reach significance
           chosen_arm_t = alg.nextpull()  # chose arm(s) to pull next
           reward_t = 1   # sample a reward for this arm, we just use 1 for simplicity here
           alg.update(chosen_arm_t, reward_t)       # update the bandit
           t+=1
```











