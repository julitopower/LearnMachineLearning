# Policy gradient agent

## CartPole

With the following configuration after 2000 eposides, it converges almost perfectly

```
# cartpole.py discount_factor learning_rage entropy_factor
taskset -c 0-3 python cartpole.py 0.99 0.00001 0.0001
```
