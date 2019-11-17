---
layout: default
title:  Home
---
![A Marvelous Minecraft Mansion](https://image.winudf.com/v2/image/Y29tLmx1Y2t5Z3VpYTk5Lm1vZGVybmhvdXNlc2Zvcm1pbmVjcmFmdF9zY3JlZW5fMF8xNTExMTc2NDMzXzAzNA/screen-0.jpg?fakeurl=1&type=.jpg)


## Introduction
One of the most beloved aspects of Minecraft is the ability to create breathtaking mansions, secret hideouts, and more. All of the costs and physics of the world no longer hold the player back. Although it is incredibly fun to conceptualize different structures to build, the design is only the beginning. The player must take on the tedious and time-consuming task of building the entire thing block by block. Here is where our project Collabuilder comes to the rescue. Imagine having a full construction squad consisting of smart AI that work together to completely build out any blueprint you provide. After doing the hard work of creating your design, you can sit back, grab a drink, and watch them get busy! On our project website, you can learn how this is possible, what kind of hurdles we had to overcome, and see how the project progressed over time. 

## The Basics
Our project is built using the Malmo Platform provided by Microsoft, which allows developers like us to create cool AI projects for Minecraft. The "brains" of our AI agent is a convulutional neural network that we used from the Keras library. We train our agent by using a method call "reinforcement learning" where we simply allow the agent to perform actions, and reward it when it does something we like, and punish it when it does something wrong. Now, to understand, how the reward system works, we need to talk a deeper look into how the agent works. We initialize a mission by placing the agent in the world and giving it a blueprint, which is simply a 3D array of different block types. The agent has a limited set of actions that it can use to build out this blueprint: "move/jump forward", "turn left", "turn right", "place block", and "remove block". The agent will analyze the current state of the environment as well as the blueprint and choose an action accordingly. If the agent places a block somewhere it doesn't belong, or meanders about aimlessly, it will be punished. On the other hand, it it quickly places blocks and fulfills the blueprint, it will be rewarded generously. We hope to train it over many trials, such that it slowly improves and becomes increasingly more competent at completing a wide array of blueprints.

## Training the agent
Training our agent to perform such a complex task is not easy. Before training, the agent has absolutely no logic encoded into it, so it performs actions completely randomly. It needs several thousands of training episodes before our reinforcement learning can actally kick in and teach the agent how to start building interesting structures. This is a lot of time, and training in Minecraft itself would make it even slower. To expedite the process, our team developed a lightweight Minecraft simulator which is much faster, and allows us to attempt training many models much more rapidly. Another technique that we have utilized to train the agents is called "Curriculum Learning". The idea is that rather than throwing grand blueprints that would be nearly impossible for the naive agent to build, we start with very simple structures and gradually increase the complexity of the tasks over time in the form of lessons. Each time the agent finshes the lesson, it has become slightly smarter and learned a new concept that will allow it to be better equiped for the next level of complexity. 

If you want to a deeper dive into our project, below are links to the source code and detailed technical reports:

### Source code: 
https://github.com/SurajSSingh/Collabuilder/

### Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)


