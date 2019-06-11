---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
<!-- Only change fields below -->
title: PPO2
summary: A simple implementation of the PPO2 model. Hidden state is computed using 2 MLP with Tanh activation function 
image: _.jpeg
author: andompesta
tags: [RL]
github-link: https://github.com/andompesta/ppo2
accelerator: "cuda"
---
```python
import torch
model = torch.hub.load('andompesta/ppo2', 'ppo2', reset_param=True, force_reload=True, input_dim=obs_size, hidden_dim=hidden_dim, action_space=action_space)
```
<!-- Walkthrough a small example of using your model. Ideally, less than 25 lines of code -->

### Model Description


### References
Original implementation: https://github.com/openai/baselines/tree/master/baselines/ppo2.
Paper: https://arxiv.org/abs/1707.06347