# Continual Dreamer Implementation 
This repository hosts the PyTorch implementation of the Continual Dreamer algorithm, a state-of-the-art reinforcement learning model introduced in the research paper ["The Effectiveness of World Models for Continual Reinforcement Learning"](https://arxiv.org/abs/2211.15944) by Kessler et al. Continual Dreamer represents a significant advancement in the field, demonstrating remarkable performance in leveraging world models for continual learning tasks.

## Algorithm Overview


The Continual Dreamer algorithm is distinguished by its innovative two-phase learning process, designed to tackle the challenges of continual learning within complex, dynamic environments. A key feature of this process is the strategic use of Reservoir sampling in the World Model Learning phase, which is crucial for the algorithm's effectiveness:

1. World Model Learning (Learning from Past Experiences): This initial phase is dedicated to constructing a robust world model from the agent's accumulated experiences. The integration of Reservoir sampling is pivotal here, as it meticulously manages the memory to preserve essential knowledge from various levels and scenarios encountered by the agent. This methodical approach to memory management is instrumental in circumventing catastrophic forgetting, a common obstacle in reinforcement learning where new learning can overwrite or disrupt valuable information from past experiences.
The use of Reservoir sampling ensures that the agent maintains a balanced and representative sample of past experiences, regardless of the volume of new data encountered. This balance is vital for sustaining the continuity of learning, allowing the agent to retain critical insights and strategies from previous levels while seamlessly integrating new knowledge. As a result, the agent is endowed with the property of continual learning, enabling it to adapt and evolve in response to the ever-changing complexities of the game environment.

2. Actor-Critic Learning (Learning by Dreaming): Following the construction of the world model, the agent embarks on the second phase, where it 'dreams' or generates synthetic experiences based on the learned model. This imaginative exploration allows the agent to refine its decision-making and strategies, bolstering its performance in real-world tasks.

By harmonizing the meticulous memory management of Reservoir sampling with the strategic foresight of the Actor-Critic learning, the Continual Dreamer algorithm equips agents with the capability to engage in perpetual, adaptive learning. This unique approach ensures that agents can navigate through diverse challenges and scenarios without the risk of losing invaluable past learnings.


## Features
* Pytorch;
* State-of-the-Art performance in continual learning;
* World Model;
* Actor Critic;
* Dreamer.

## Installation
1. Clone the repository
   ```
   git clone https://github.com/LinYyou/ContinualDreamer
   ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
## Running the Model
- for training
  ```
  python main.py -t
  ```
- for testing
  ```
  python main.py -e --render
  ```  
## References
- [The Effectiveness of World Models for Continual Reinforcement Learning;](https://arxiv.org/abs/2211.15944)
- [Mastering Atari with Discrete World Models;](https://arxiv.org/abs/2010.02193)
- [Planning to Explore via Self-Supervised World Models.](https://arxiv.org/abs/2005.05960)
