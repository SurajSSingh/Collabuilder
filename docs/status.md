---
layout: default
title:  Status
---

## Project Summary: 

Our project is to train a Minecraft agent to be able to build 3-dimensional structures based on a blueprint that is input as a 3-dimensional array of block types. We will train the agent using a neural network and reinforcement learning. We will use neural network models from the keras library. Furthermore, we will also use curriculum learning to allow the agent to be trained more efficiently and effectively. Ideally, the agent should be able to build the input blueprint with perfect accuracy, as fast as possible. Initially we had hoped that the agent would be able to construct ceilings for the buildings, but our current action space makes this near impossible, so we are taking this goal off the table for now. We have developed a restricted Minecraft simulator that emulates the agents actions, which allows us to train the agent much more quickly by avoiding the overhead of running Minecraft. Another subgoal that we currently have is to develop a model tester, that will allow us to automate the process of trying different curriculums and hyper-parameters for our neural network. Ideally, we will scale our project up to multiple agents, but for now we are focused on getting one agent to perform very well. Also, we will attempt to add more complexity to the blueprint through different materials and structure types if time allows.


## Approach:

At a high level, we implemented a convolutional neural network that accepts a blueprint and a global world state, and produces Q-values for the available actions. This network was trained using curriculum reinforcement learning, with a double Q-learning update policy.

In particular, the neural network accepts input tensors with shape $$ (2, B, W, H, L) $$, where $$B$$ is the number of block types being used, plus one type each for “air” and “agent”, and $$ W, H, L $$ are the width, height, and length of the arena. The input is a map of the world, as a 3D array of the blocks and agent in the world encoded categorically (four dimensions in total), and another array of the same dimension describing the “blueprint”, or desired state of the world.
The network first applies two levels of 3-D convolutions, each using 8 cubic filters, 3 units to a side. Then, a layer of max-pooling is applied, followed by two fully-connected layers. The output is a Q-value estimate for each of five available actions: Turn left, Turn right, Jump/Move forward, Place block, and Remove Block.
For training, a double-Q-learning update policy is implemented, whereby a copy of the network with frozen weights $$ \Theta_0 $$ is used to estimate the target Q-values $$ \hat{Q} $$ as follows:

$$ \hat{Q}(s, a) = r + \gamma \max_{a \in A}Q(s’, a ; \Theta_0) - Q(s, a; \Theta) $$

Where $$r$$ is the reward for taking action $$a$$ in state $$s$$, $$ \gamma $$ is a discount factor (which we set to 0.95), $$s’$$ is the resulting state, and $$\Theta$$ is the weights on the training network. This target value is then used for one iteration of ``supervised’’ learning for the target network, to update $$\Theta$$. Every 20 iterations, $$ \Theta_0 $$ is set to the current value of $$ \Theta $$.

While running, actions are chosen by an epsilon-greedy policy, with an exponentially decaying epsilon. Initially, there is a $$ \varepsilon_0 = 0.5 $$ chance that the agent will ignore its Q estimates and choose an action at random. Every episode, $$ \varepsilon $$ is multiplied by a decay factor $$ \gamma_{\varepsilon} $$, which is calculated to yield a final $$ \varepsilon_f = 0.01 $$, by $$ \gamma_\varepsilon = (\varepsilon_f / \varepsilon_0)^{1/N} $$, where $$ N = 1500 $$ is the expected number of episodes.

Since the task of building structures requires a complex sequence of actions to be correctly executed, we implemented a curriculum learning system. It has a sequence of lessons, functions which create constrained versions of the full problem, which progress in difficulty. Associated with each lesson is a target reward. If the agent achieves at least the target reward for 50 consecutive episodes, the system progresses to the next lesson.

As another way to accelerate learning, we implemented a shaped reward function, which yields large rewards for placing blocks correctly and large punishments for leaving the arena, but also yields small rewards for moving closer to where blocks are needed and for facing an incomplete block. This helps to guide behavior during the early stages of learning.

As a final major improvement to our training speed, we implemented a simulation of the aspects of Minecraft relevant to our project. This simulation runs many times faster than the Malmo & Minecraft stack, which enables training models with thousands of episodes in reasonable time.


## Evaluation: 

### Quantitative

At this stage, the most important evaluation metric is the progress the agent has made through the curriculum. Currently, our best agent has passed ?? lessons, up to the ?? lesson. A closely related metric is the number of episodes it takes the agent to pass each lesson. Currently, our best agent passes the first lesson in ?? episodes, the second lesson in  ?? episodes, etc. 

Another important metric is the accuracy with which the agent is able to construct the blueprint. We can easily calculate the amount of correct and incorrect blocks placed by the agent by comparing the minecraft world state with the blueprint. It is very important to us that our agent is not placing blocks where they do not belong, and following the input blueprint perfectly. Achieving and maintaining perfect accuracy with increasing complexity of models is a key metric when evaluating the success of our project. Currently, we demand the agent place blocks with perfect accuracy to pass the first and second lessons (check this fact), so we know the 

### Qualitative

Although the agent may be able to quickly and perfectly build a blueprint that is simply 2 blocks in a line, this is not all that impressive. The complexity of the blueprints that our agent is able to build is another key metric that will determine the success of our project. Building structures that are tall with random walls would be the current goal. Since the curriculum slowly increases the complexity of the blueprints, it is basically a quantitative way to gauge this metric.

When we simply look at the agent building the blueprint, does it seem “smart”. Does it walk around aimlessly before arriving at the next building area or does it effectively and gracefully navigate the structure while building it. Is the agent able to create stair structures to reach high areas, or does it give up and consider certain areas impossible to reach after a while? Does the agent make sure that it leaves space to walk, or does it wall itself in? These are all mistakes that a real Minecraft player would think are silly, but are perfectly possible for an agent that is learning with no preexisting knowledge. Just by watching the agent build, it is easy to get a feel for whether it is up to standards, or it is not quite as smart as it should be.


## Remaining Goals and Challenges:

So far, we have worked to develop the framework for our AI, but we need to do a lot more in order to get it to the desired level. We currently have lessons developed for curriculum learning, a simulation for fast training, and a neural network integrated with our training mechanism. Moving forward we need to leverage these resources to get our agent to perform the desired task well. Although we have gotten the agent through some of the basic lessons, we still need the agent to be able to build structures with more height. One difficult task we anticipate when getting to structures is definitely building upwards. It will take a lot of careful training to teach the agent to build tall sequences of blocks without falling off. We are hoping that a cleverly developed curriculum will allow us to teach the agent to build taller structures.

Another goal is finding better configurations for our neural network that will enable the agent to build increasingly complex blueprints. We will definitely need to further research convolutional networks to gain a deeper understanding on how to optimize for our specific problem. Also, we plan to develop a model testing framework that will utilize our existing training framework in order to expedite our search for better curriculums and better architectures for the network. Because there are many places in which our system can be modified, it is difficult to say what configuration will work best. Currently, we have been manually running the training overnight, and it takes a long time to figure out what works well, and where the problems lie. We hope to automate this process by using a model tester into which we can input a list of different configurations that we want to test, and have it try them all out. Furthermore, we will take the best performers from the first couple of lessons and have them move on to the next round of lessons. Ideally, this would allow us to narrow down the best set of parameters through efficient testing that would be cumbersome to perform manually.

As made evident with our team name “Collabuilder”, we would love to be able to introduce multiple agents into the environment and have them work together in order to build the structures. Our simulator would be all the more crucial in this setup, and it would be difficult to find the correct way to train agents simultaneously. Given time, this would be our final addition to the project.


## Resources Used:

Keras is used for our neural network model, with Tensorflow as its backend. The Minecraft simulator relies heavily on Numpy. Plotting and displays use Matplotlib.

The paper “A Guide to Convolution Arithmetic for Deep Learning.” by Dumoulin and Visin gave us some practical explanation of convolutional layers. In particular, this paper explains how convolutional layers change the shape of their input tensors.

