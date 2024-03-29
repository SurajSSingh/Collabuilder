---
layout: default
title:  Final Report
---
## Video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/Y2gcbZS6NvI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Project Summary:

Our project is to train a Minecraft agent to be able to build 3-dimensional structures based on a blueprint that is input as a 3-dimensional array of block types. We trained the agent using a neural network and reinforcement learning, built using the Keras library. We also used curriculum learning to allow the agent to be trained more efficiently and effectively. We evaluated the agent based on its completeness, accuracy, and efficiency in building the blueprint. Initially we had hoped that the agent would be able to construct ceilings for the buildings, but our current action space makes this near impossible, so we removed this goal. We had also hoped to implement multiple agents, but were forced to remove this goal due to time constraints and the difficulty of getting a single agent to perform well.

We developed a simple baseline agent that uses a traditional breadth first search approach to construct a blueprint as a comparison point. We wanted to find out whether reinforcement learning and neural networks could measure up to the traditional agent when constructing the blueprints. Although traditional BFS is a great solution when dealing with simple blueprints,the code becomes increasingly more complex as more factors like height, shape, and block types are added in. The key motivation here is to seek a more general agent model that scales better that the traditional model by requiring more training time rather than more code when the complexity is increased.

We have developed a restricted Minecraft simulator that emulates the agents actions, which allows us to train the agent much more quickly by avoiding the overhead of running Minecraft. We also developed a model tester, which ran many models and curriculums as a batch to do hyper-parameter optimization. Using these tools, we were able to create an agent that very successfully builds one-layer blueprints, and makes some progress on multi-layered ones. We also experimented with a Dueling DQN agent.




## Approach:

At a high level, we implemented a convolutional neural network that accepts a blueprint and a global world state, and produces Q-values for the available actions. This network was trained using curriculum reinforcement learning, with a double Q-learning update policy. We also implemented a baseline agent using traditional AI techniques that served as a comparison point for us to gauge the success of the models that we built.

In particular, the neural network accepts input tensors with shape $$ (2, B, W, H, W) $$, where $$B$$ is the number of block types being used, plus one type each for “air” and “edge” (positions outside the arena), and $$ W $$ and $$ H $$ are the width and height of the observation, centered on the agent’s position. The input is a 3D array of the blocks in the world encoded categorically (four dimensions in total), and another array of the same dimension describing the “blueprint”, or desired state of the world.

In order to efficiently research architectures and hyper-parameters for our agent, we developed a model-testing framework. This framework accepts a directory of configuration files, and runs the models described by those files in rounds, one lesson at a time. It automatically detects and weeds out models that fail to train correctly, and collects statistics for later analysis, to aid in selecting the highest-performing models, as depicted in the diagram below:

<p align="center">
    <img src="images/Model_Tester.png" title="Model Tester">
</p>

As another way to accelerate learning, we implemented a shaped reward function, which yields large rewards for placing blocks correctly and large punishments for leaving the arena, but also yields small rewards for moving closer to where blocks are needed and for facing an incomplete block. This helps to guide behavior during the early stages of learning.

As a final major improvement to our training speed, we implemented a simulation of the aspects of Minecraft relevant to our project. This simulation runs many times faster than the Malmo & Minecraft stack, which enables training models with thousands of episodes in reasonable time. We also built a visualization of the agent in this simulated model. Here, the red voxel is the agent, light blue voxels are blueprinted blocks that haven't been placed, and dark blue blocks are real blocks in the world.

<p align="center">
    <img src="images/big_arena.png" width="500" height="425" title="Simulation Display">
</p>

All successful versions of our agent first applied one or more convolutional layers, followed by fully connected layers. Using the model-testing framework we developed, we made the following comparisons:

* Full-world vs. agent-centric observations: We tried giving the agent the entire world as an observation, with a special "block" type for the agent's location, versus giving the agent a fixed-size observation centered on the agent's location, without explicitly marking the agent's location. The latter trained more reliably and robustly.

* Convolutional layer arrangements: We tried 1-3 layers of convolution, with varying numbers of filters, with and without padding. We found that 2 layers, with 16 and 8 filters respectively, and without padding, was optimal.

* Post-convolution layer: We tried max-pooling with various stride lengths, batch-normalization, and nothing between the convolutions and fully-connected layers. We found that directly connecting the last convolution to the fully-connected layers was most effective.

* Fully-connected layers: We tried 1 to 4 fully-connected layers, with 4 to 64 neurons in varying arrangements. We found that 3 layers, with 32, then 16, then 4 neurons, was optimal.

* Activation function: We tried using ReLU, Tanh, and ELU (Exponential Linear Unit) activation functions on hidden layers. Of these, ELU performed best.

* Curriculums: We tried varying the number and kinds of lessons in the curriculum, with more lessons and smaller incremental goals, versus fewer lessons with larger goals. We found that having just a few lessons, with large goals and long training times, was most effective. Having many small, incremental goals led to less effective transfer of the training between lessons. We suspect the agent overfit to the narrowly tailored lessons.

Using the results described above, our best network first applies two levels of 3-D convolutions, first using 16 filters, then using 8 filters. Then, we apply three fully-connected layers, numbering 32, 16, and 4 neurons, with all hidden layers using ELU activations. The output is a Q-value estimate for each of four available actions: Turn left, Turn right, Jump/Move forward, and Place block.

For training, a double-Q-learning update policy is implemented, whereby a copy of the network with frozen weights $$ \Theta_0 $$ is used to estimate the target Q-values $$ \hat{Q} $$ as follows:

$$ \hat{Q}(s, a) = r + \gamma \max_{a' \in A}Q(s’, a' ; \Theta_0) - Q(s, a; \Theta) $$

Where $$r$$ is the reward for taking action $$a$$ in state $$s$$, $$ \gamma $$ is a discount factor (which we set to 0.95), $$s’$$ is the resulting state, and $$\Theta$$ is the weights on the training network. This target value is then used as the labels for one iteration of "supervised" learning for the target network, to update $$\Theta$$. Every 20 iterations, $$ \Theta_0 $$ is set to the current value of $$ \Theta $$.

While running, actions are chosen by an epsilon-greedy policy, with an exponentially decaying epsilon. Initially, there is a $$ \varepsilon_0 = 0.8 $$ chance that the agent will ignore its Q estimates and choose an action at random. Every episode, $$ \varepsilon $$ is multiplied by a decay factor $$ \gamma_{\varepsilon} $$, which is calculated to yield a final $$ \varepsilon_f = 0.01 $$, by $$ \gamma_\varepsilon = (\varepsilon_f / \varepsilon_0)^{1/N} $$, where $$ N $$ is the expected number of episodes for the current lesson.

Since the task of building structures requires a complex sequence of actions to be correctly executed, we implemented a curriculum learning system. It has a sequence of lessons, functions which create constrained versions of the full problem, which progress in difficulty. Associated with each lesson is a target reward. If the agent achieves at least the target reward for 15 consecutive episodes, the system progresses to the next lesson. After testing several different curriculum styles, including many that used simpler lessons to start with, we found the following curriculum most effective:
1. 10 blocks scattered randomly throughout the arena, with a randomly placed agent
2. Full blueprint, 2 levels deep
3. Full blueprint, 4 levels deep

<p align="center">
    <img src="images/Curriculum Plan.png" width="200" height="200" title="Curriculum">
</p>

It should be noted that the full blueprints are generated randomly, so the agent is not being trained on a single, fixed blueprint, but rather is expected to construct any blueprint given to it. Due to time constraints, however, we were unable to fully train a model on the full arena size using this curriculum. The resutls of our partially trained model are summarized below.

After the status report, one option of improvement we considered was to change the architecture of the model to support a dueling DDQN structure. The main difference with this and our current model (regular Double DQN) was that there would be a separate calculation of the state of the agent and the actions it could take in that state. This separation would then be merged back as state reward plus the advantage (we calculated as each action’s reward minus the total average reward of all actions).
<p align="center">
    <img src="images/Equation.png" width="600" height="90" title="Dueling Equation">
</p>
The equation here means as follows: V(s) is the value of the state, A(s,a) is the value of the action advantage (how good is the best action) at a given state and A(s,a') is the action advantage for all actions a'.  

What Dueling DQN means for us is that the model would learn which states are considered good (e.g. in front of a block to place) and which actions are considered good (e.g. place block) independently. We predict that this would make the training much smoother with little trade-off in time (as the underlying convolution architecture remains the same). In practice, the Dueling DDQN model did perform much better than the plain DDQN in terms of accuracy of blocks and mission time, but it did take longer to learn up to the DDQN. The model was also able to much more easily handle complex lessons in the curriculum, which means that it was able to generalize much more efficiently than the other model. With more time to tweak the setting, we may have fully switched over to Dueling DQN.


Lastly we built a simpler agent that used traditional techniques to construct the blueprint. This agent worked perfectly and efficiently, but required us to write code for all cases, which is far more difficult to scale if we wanted to add new features to the blueprints. The logic for this agent is as follows:

    Initialize starting position and save all non-air blocks in blueprint in a set(unplaced_blocks)
    While(unplaced_blocks is not empty):
        breadth first search from current position to find an unplaced block
        navigate to block and place it
        remove block from unplaced_blocks


## Evaluation:

### Quantitative
Due to the excessive (>24 hour) training time, we were unable to test a fully trained agent. The metrics reported below are for the partially trained version of our best agent, which had completed the first lesson, and was approximately halfway through the second lesson, of its curriculum.

Since our agent wasn’t able to completely build the full blueprint, the most important quantitative metric is the percent completion of the blueprint at the end of the mission. Currently, our best agent completes around 40% of a full blueprint when using a purely greedy strategy, but completes 60-90% when using the epsilon-greedy strategy during training. This disparity is due to the pure-greedy strategy getting “stuck” in situations where it reacts inappropriately, for instance by attempting to place a block it has already placed. While the pure greedy strategy then gets exactly the same input, and cannot progress, the epsilon-greedy strategy can randomly move out of this situation. The baseline agent can always achieve 100% completion of the blueprint, so our RL agent was slightly inferior by this metric.

Closely related to this metric is the reward the agent earns per episode. This was captured during training, and is displayed below, along with the length of each episode. The dashed vertical lines denote where the agent moved from one lesson to the next. Also note that these graphs are smoothed to avoid clutter and more clearly show trends in the data.

<p align="center">
    <img src="images/Reward-Plot.png" width="600" height="400" title="Reward Plot">
</p>

<p align="center">
    <img src="images/Length-Plot.png" width="600" height="400" title="Length Plot">
</p>

Observe how the agent initially receives mostly negative rewards, as it is continually punished for leaving the arena and placing superfluous blocks. Over the course of the first lesson, it slowly learns how to navigate and place blocks effectively, eventually reaching near-optimal rewards for the first lesson. The rewards drop at the start of the next lesson, primarily because the epsilon value used for epsilon-greedy action selection is reset to $$\varepsilon_0 = 0.8$$. We see a corresponding drop in the episode length at this time, as the agent randomly chooses actions that often lead to ending the mission prematurely. At around 4500 episodes, we see a sharp increase in the rewards received for the second lesson, as well as a levelling off of episode length near the 30 second maximum length. This is when the agent first becomes effective at placing blocks in a multi-level blueprint, and the steady increase in reward thereafter is attributed to increasingly effective strategies for building, and decreasing epsilon values. At around 5500 episodes, we noticed the time limit of 30 seconds was insufficient to complete the episode, even with optimal behavior, and that the agent nearly always times out, showing that it is exploiting the available time to its fullest capacity. To remedy this, and because we didn’t have time to retrain an agent from scratch, we built a new lesson using the same two-level blueprints, but with a much higher (5 minute) time limit, and manually moved the agent on to this new lesson. Once again, the reset in epsilon caused a sharp decrease in reward and episode length. However, the agent soon exploits the longer episode time, as we see the episode length sharply exceed the previous maximum, and use the behaviors it learned in previous lessons to place large amounts of the blueprint. We expect that further training would continue the upward trend in the reward graph, and would reduce the variance seen in this later stage. Based on prior testing, we also expect that sufficient training would eventually decrease the episode lengths slightly, as the agent optimizes its behavior for speed.

### Qualitative

In our status report, we mentioned that the complexity of blueprints was a key qualitative metric of our agent’s ability. Since our agent is able to construct 2-layer blueprints with moderate competency, as explained above, we have a distinct improvement from the status report, in which the agent placed only a single block, and are close to meeting the goal of a fully-sized blueprint without a roof. We expect that the current agent could meet this goal without modification, given sufficient training time. An example of the kind of blueprint our agent attempted is given below, displayed using our simulation. Notice that there is a solid foundation layer, and walls both around the exterior and throughout the interior separating various rooms in the structure.

<p align="center">
    <img src="images/Level-2.png" width="600" height="510" title="2-Level Blueprint">
</p>

This blueprint is close to the full goal of a “tall” (4-layer) blueprint, in which all the walls have been extended vertically, like below:

<p align="center">
    <img src="https://drive.google.com/uc?id=1CrQ6ele3FRX3bp80vaEz-9eAM-eNxiRq" width="600" height="510" title="4-Level Blueprint">
</p>

Another key qualitative metric is to evaluate whether the agent behaves "reasonably" to a human observer. On this metric, the agent scores very highly on the first lesson, and moderately on the second. When given randomly scattered blocks on the ground to place, the agent effectively navigates to positions where blocks are needed and places them, and even explores the world effectively when there are no incomplete blocks in its field of view. The first lesson is comparable to the baseline agent which seems intelligent because it is hard coded to navigate with minimum use of moves and choose the closest possible block to place. On the two-layer blueprints, the agent tends to build walls by constructing stairs to begin the wall, then walking along the top of the wall and constructing the next step as it goes. In this respect, the agent behaves highly reasonably. However, the agent tends to get “confused” by the interior walls, eventually attempting to place a block that has already been placed, or turning around indefinitely. In this respect, the agent fails to behave reasonably. However, given the current level of training, we consider this behavior to be acceptable. Here, our agent falls behind the baseline. The baseline cannot be “confused” and thus, will never hit a case where it spins in circles or repeatedly places a block.

Another metric that we gauged was accuracy. When watching the agent and the blueprint display, it is clear that the agent sometimes places superfluous blocks that are not needed. We penalized this behaviour, but the agent would still sometimes place the blocks in the wrong place. Raising the penalty for this behaviour improved accuracy, but made it more difficult for the agent to complete the full blueprint. Although the baseline model has perfect accuracy, our fully trained mdoels place very few superfluous blocks when trained with high penalties, so the performance is comparable.

### Summary of Results

Overall, we consider this to be a moderately successful project. While the baseline agent using traditional AI techniques out-performed our RL agents, we were able to construct an RL agent that makes nontrivial progress towards the goal. In doing so, we have highlighted the strengths of an RL approach, in not having to explicitly code information about the problem into the agent, as well as some weaknesses, in particular the difficulty of training an agent that can "plan ahead" effectively. Despite its imperfections, we believe our agent shows significant progress towards our goal of constructing complex structures in Minecraft.

## Resources Used:

Keras is used for our neural network model, with Tensorflow as its backend. The Minecraft simulator relies heavily on Numpy. Plotting and displays use Matplotlib.

The paper “A Guide to Convolution Arithmetic for Deep Learning.” by Dumoulin and Visin gave us some practical explanation of convolutional layers. In particular, this paper explains how convolutional layers change the shape of their input tensors.

The github page https://yilundu.github.io/2016/12/24/Deep-Q-Learning-on-Space-Invaders.html by Yilun Du was instrumental in getting the Dueling DQN working.

Documentation pages for all the packages mentioned here were crucial, as were the tutorials for Malmo. More general help like StackOverflow posts on relevant questions were also very helpful, but are too numerous to list here, and no one of them was important enough to deserve an individual callout. Finally, the Towards Data Science series on Medium.com was often helpful in understanding the CNN & RL topics that we were unfamiliar with.
