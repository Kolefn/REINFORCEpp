# REINFORCEpp
> A library for building and training reinforcement learning models.

![](https://kolenunley.com/portfolio/img/reinforcepp.gif)

There are many RL libraries out there that are much larger and more advanced than this one. This project does not seek to compete. Creating this library was a fantastic way to learn the detailed mechanics of RL and ANN backpropogation.


## Installation

```sh
git clone https://github.com/Kolefn/REINFORCEpp.git
```
Manual: 

1. Download
2. Unzip


## Usage example

The Deep-Q Learning agent class is a pre-built model included in the library. It is suitable for most simple tasks.
```c++
  #include "DQNAgent.h"
```
Define the agent params.

```c++
 agent_options opts;
 opts.NUM_STATE_FEATURES = 4;
 opts.MAX_NUM_ACTIONS = 4;
 opts.NUM_HIDDEN_UNITS = 100;
 opts.experience_add_every = 25;
 opts.EXPERIENCE_SIZE = 5000;
 opts.learning_steps_per_iteration = 10;
 opts.tderror_clamp = 1.0;
 opts.gamma = 0.75;
 opts.epsilon = 0.2;
 opts.alpha = 0.01;
```
Create the agent

```c++
  DQNAgent agent = DQNAgent(opts);
```

Within your main loop, where state is a float vector of size ``NUM_STATE_FEATURES``. 

```c++
  int action = agent.act(state);
```


After that, also within your main loop, where ``newState`` is a new float vector based on the new state after the ``action`` is taken.
```c++
  float reward = getReward(newState);
```

Finally, within your main loop
```c++
  agent.learn(reward);
```

Repeat.

## Credits

Andrej Karpathy - [REINFORCE.js](https://github.com/karpathy/reinforcejs)

## Meta

Kole Nunley – [Website](https://kolenunley.com)

Distributed under the MIT license. See ``LICENSE`` for more information.

[My GitHub](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/Kolefn/REINFORCEpp/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request :D
