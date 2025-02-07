# Racing-car_Actor_critic
we use actor critic algorithm implement in gym-racing car environment


# Reinforcement Learning Methods Classification

## Comparison of Different RL Methods

| Method Type | Representative Algorithms | Characteristics | Advantages | Disadvantages |
|------------|---------------------------|----------------|------------|--------------|
| **Value-Based** | Q-Learning, DQN | Learns a Q-value function $Q(s, a)$ | Model-free, suitable for discrete action spaces | Not suitable for continuous action spaces |
| **Policy-Based** | REINFORCE | Directly learns a policy function $ \pi(a / s) $ | Suitable for continuous action spaces, no need to store Q-values | High variance, slow convergence |
| Combinded method |  **Actor-Critic**  | Combines value function $ Q(s, a) $ with policy function $ \pi(a / s) $ | More stable training, reduces high variance | More complex to design |


## What is actor critic?
![image](https://hackmd.io/_uploads/rJC2V8Xtyx.png)
![image](https://hackmd.io/_uploads/r1FB7D7K1e.png)

## How actor critic implement in gyn- racing car

### gym racing car environment

```python
    env = gym.make("CarRacing-v1", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    env.reset()

```


```
actions:
    方向盤轉向（Steering）：範圍 $[-1, 1]$，負值為左轉，正值為右轉。
    油門（Gas）：範圍 $[0, 1]$，表示加速。
    煞車（Brake）：範圍 $[0, 1]$，表示減速。
states:
    輸出為一個 RGB 圖像，大小為 96x96x3，表示賽車場景的視角。
每個像素值的範圍是 $[0, 255]$，
```

```python
# 環境交互示例
import numpy as np

state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  # 渲染環境
    action = env.action_space.sample()  # 隨機採樣動作
    next_state, reward, done, info = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)
env.close()


```



## Actor critic implement in python

* Why we use cache memory: we learn a batch of data onece, so we need to store ()
* The agent class has following function:
    * choose action
    * cache memory
    * learn-> 

(s,a,s',r,done)

**learn critic->goals accurate predict the expected reward of environment : input:state,output:value**

minimize : true(env)_ expected_reward- critic_predict_value

=y-critic(s)

y=r+gamma* critic(s')

minimize loss function=loss(y,critic(s)) -> gradient descent

critic(theta+= slope of loss function to theta)


actor -> **goals: recieve maximum reward of output action**

s->actor_net->a->critic(s')

maximum critic(s')->

actor_net(theta)<-critic(s')*(slope of actor net to every theta)


