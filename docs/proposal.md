---
layout: default
title: Proposal
---

## Summary:
Our project is a multi agent AI with multiple builders working together to construct a house. We plant to have each individual builder use a neural network trained through reinforcement learning and curriculum learning. We anticipate uisng the Keras library in order to achive this. The input will be a 3D layout of a floor plan of a house, that we can depict using a 3 dimensional array with the necessary blocks in the desired location. If successful, our agents will work together and build the input floor plan perfectly. Furthermore, if we achieve this goal, we will try and make them faster, and see how they can learn to optimize different building tasks. We plan to do one configuration where the agents are trained for separate specialized tasks, and another where there are no specialties, and they may learn to specialize by themselves, or find a more effective solution.

## AI/ML Algorithm
* Reinforcement-Learning to Train Neural Network
* A* search (for navigation)
* Curriculum Learning

## Evaluation Plan
The first, and most important metric will be accuracy. We can easily measure this by comparing the state of the environment with the desired floorplan. The percentage of accuracy will be a great evaluation metric. Another metric that wil become important once the agnets are accurate is speed. The baseline accuracy will be 0 since the agent will have no idea how to build a structure corresponding to a structure. The speed will also be very low. We will need to experiment with a single agent first before moving onto a multiagent situation. An initial sanity test will be trying to build a simple room with a single agents, and then with multiple agents. 
* Individual:
	* Being able to place the correct block
	* Being able to accomplish their specific task (e.g. being able to place the foundations of the house)
* Team: 
	* Qualitative evaluation of whether they are actually collaborating or just doing their own thing and no interfering with each other (i.e. are they communicating with each other effectively)
	* Being able to construct any house given a floor plan of varying complexity

## Appointment
Wednesday October 16, 10:10 - 10:30pm
