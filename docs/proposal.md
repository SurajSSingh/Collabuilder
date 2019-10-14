---
layout: default
title: Proposal
---

## Summary:
Our project is a multi agent AI with multiple builders working together to construct a house. We plan to have each individual builder use a neural network trained through reinforcement learning and curriculum learning. We anticipate uisng the Keras library in order to achive this. The input will be a 3D layout of a floor plan of a house, that we can depict using a 3 dimensional array with the necessary blocks in the desired location. If successful, our agents will work together and build the input floor plan perfectly. Furthermore, if we achieve this goal, we will try and make them faster, and see how they can learn to optimize different building tasks. We plan to do one configuration where the agents are trained for separate specialized tasks, and another where there are no specialties, and they may learn to specialize by themselves, or find a more effective solution.

## AI/ML Algorithm
* Reinforcement-Learning to Train Neural Network
* A* search (for navigation)
* Curriculum Learning

## Evaluation Plan
The first, and most important metric will be accuracy. We can easily measure this by comparing the state of the environment with the desired floorplan. The percentage of blocks from floorplan built will be a great evaluation metric. We will probably start with extremely easy cases to get right involving a single block, and work up to more difficult cases.  When learning, we will grant rewards for the agents whenever they build a block accurately, and penalize them for building blocks unnecessarily. Another metric that will become important once the agents are accurate is speed. The baseline accuracy will be 0 since the agent will have no idea how to build a structure corresponding to a plan. The speed will also be very low. An appropriate baseline could be determined by having a human build a floorplan, and seeing the difference in speed. This would allow us to understand whether the AI actually has any practical use. We will need to experiment with a single agent first before moving onto a multiagent situation. We hope to be able to reach a very high accuracy, as well as decent speed. 

Qualitatively, we hope that the agents are able to collaborate in a meaningful way. An initial sanity test will be trying to build a simple room with a single agents, and then with multiple agents. We can then move on to more and more advanced floor plans. We are very curious to see if the agents will be able specialize without any external assistance. I think it will be relatively clear to see how the algorithm is functioning internally by following each individual agent and tracking their behaviour. Obviously if one agent builds the whole structure while the other two stand idle, the muti-agent aspect is broken, even though accuracy is 100%. Our moonshot case would be many agents seamlessly working together and building any floorplan far more efficiently than a group of humans possibly could. Currently, we think maybe one agent will build floors, one will build walls, and one will do the ceiling; we are curious to see if they do something totally different. Maybe they would start at opposite corners and build into each other, etc.


## Appointment
Wednesday October 16, 10:10 - 10:30pm
