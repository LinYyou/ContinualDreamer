import os
import io
import uuid
import glob
import pickle
import random
import datetime
import collections
import numpy as np


class ReplayBuffer():
    def __init__(
            self,
            capacity=1e6,
            minimum_length=5,
            maximum_length=50,
            prioritize_endings=False,
            path = "episodes",
            recent_threshold=0.5,
            reservoir_sampling = True
    ):
        # Settings
        self.capacity = capacity
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        self.prioritize_endings = prioritize_endings

        self.task_idx = 0

        # Create the directory for storing the episodes if it doesn"t exists
        self.path = path
        if not os.path.exists(path): os.mkdir(path)

        # Load buffers if it"s not a new run
        self.completed_episodes, self.tasks, self.episode_rewards = self.load_episodes(
            self.path,
            self.capacity
        ) 

        # Counters
        self.ongoing_episode = collections.defaultdict(list)
        self.total_episodes, self.total_steps = self.count_episodes(self.path)
        self.loaded_episodes = len(self.completed_episodes)
        self.loaded_steps = sum(self.episode_length(ep) for ep in self.completed_episodes.values())

        # Reservoir sampling and 50:50 sampling
        self.reservoir_sampling = reservoir_sampling
        self.recent_threshold = recent_threshold
        
    @property
    def stats(self):
        stats = {
            "total_steps": self.total_steps,
            "total_episodes" : self.total_episodes,
            "loaded_steps" : self.loaded_steps,
            "loaded_episodes" : self.loaded_episodes,
            "average_task" : np.mean([value for _, value in self.tasks.items()])
        }
        return stats
    
    def add_step(self, transition):
        episode = self.ongoing_episode
        for key, value in transition.items():
            episode[key].append(value)
        if transition["is_last"]:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):

        # Skip the episode if too short
        episode_length = self.episode_length(episode)
        if episode_length < self.minimum_length:
            print(f"Episode it's too short, skipping it..")
            return
        
        # Update counters
        self.total_steps += episode_length
        self.loaded_steps += episode_length
        self.total_episodes += 1
        self.loaded_episodes += 1
        
        # Retrieve episode informations
        task = self.task_idx
        episode = {key: self.convert(value) for key, value in episode.items()}

        # Save the episode locally
        filename = self.save_episode(self.path, episode, task, self.total_episodes)

        # Add the episode to the buffers with filename as key
        self.completed_episodes[str(filename)] = episode
        self.tasks[str(filename)] = task
        self.episode_rewards[str(filename)] = episode["reward"].astype(np.float64).sum()
        
        # Reservoir Sampling
        if self.reservoir_sampling:
            if self.loaded_steps >= self.capacity:

                # Sample a random episode from the buffer
                idx = np.random.randint(self.total_episodes)

                # Remove it with probability current_size/max_size
                if idx < self.loaded_episodes:
                    filenames = [key for key, _ in self.completed_episodes.items()]
                    filename_to_remove = filenames[idx]
                    episode_to_remove = self.completed_episodes[str(filename_to_remove)]
                    self.remove_episode(filename_to_remove, episode_to_remove)
        
        # Remove random episodes until the buffer is no more full
        while self.loaded_episodes > 1 and self.loaded_steps > self.capacity:
            # Reservoir sampling: randomly choose a saved episode
            if self.reservoir_sampling:
                filename_to_remove, episode_to_remove = random.sample(self.completed_episodes.items(), 1)[0]
            # Otherwise remove the oldest one
            else:
                filename_to_remove, episode_to_remove = next(iter(self.completed_episodes.items()))
            self.remove_episode(filename_to_remove, episode_to_remove)

        # Reservoir sampling: locally save the buffer
        if self.reservoir_sampling:
            with open(f"{self.path}/rs_buffer.pkl", "wb") as file:
                pickle.dump(list(self.completed_episodes.keys()), file, protocol=pickle.HIGHEST_PROTOCOL)
        
    def remove_episode(self, filename_to_remove, episode_to_remove):

        # Update counters
        self.loaded_steps -= self.episode_length(episode_to_remove)
        self.loaded_episodes -= 1

        # Remove the sampled episode from the buffers
        del self.completed_episodes[str(filename_to_remove)]
        del self.tasks[str(filename_to_remove)]
        del self.episode_rewards[str(filename_to_remove)]

    def save_episode(self, path, episode, task, total_episodes):
        # Save episode with its info
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        id = str(uuid.uuid4().hex)
        episode_length = self.episode_length(episode)
        filename = f"{path}/{timestamp}-{id}-{task}-{episode_length}-{total_episodes}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with open(filename, "wb") as f2:
                f2.write(f1.read())
        return filename

    def load_episodes(self, path, capacity):

        # Retrieve all saved episodes
        filenames = sorted(glob.glob(f"{path}/*.npz"))

        # If we have a maximum capacity and we already have a buffer saved, override the episodes
        if capacity:
            num_steps = 0
            num_episodes = 0
            if os.path.exists(f"{path}/rs_buffer.pkl"):
                with open(f"{path}/rs_buffer.pkl", "rb") as file:
                    filenames = pickle.load(file)
            
            # Take only the most recent episodes we can store
            for filename in reversed(filenames):
                episode_length = int(str(os.path.basename(filename)).split("-")[3])
                num_steps += episode_length
                num_episodes += 1
                if num_steps >= capacity:
                    break
            filenames[-num_episodes:]

        episodes = {}
        tasks = {}
        episode_rewards = {}

        # Retrieve the episodes information and save it in the buffers
        for filename in filenames:
            with open(filename, "rb") as file:
                episode = np.load(file)
                episode = {key: episode[key] for key in episode.keys()}

            episodes[str(filename)] = episode
            task = int(str(os.path.basename(filename)).split("-")[2])
            tasks[str(filename)] = task
            episode_rewards[str(filename)] = episode["reward"].astype(np.float64).sum()

        return episodes, tasks, episode_rewards

    def sample(self, batch_size, length):
        batch = []
        sequence = self._sample_sequence()

        while len(batch) < batch_size:
            # Each sample in the batch must have the same length, but the sample may be longer or shorter
            # so we add chunks of it until we reach the desired length, and resample if we consume all the sequence
            sample = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {key:value[:needed] for key, value in sequence.items()} 
                remaining = {key:value[needed:] for key, value in sequence.items()}
                for key, value in adding.items():
                    sample[key].append(value)
                added += len(adding["action"])
                if len(remaining["action"]) < 1:
                    sequence = self._sample_sequence()
            sample = {key: np.concatenate(value) for key, value in sample.items()}
            batch.append(sample)
        
        #batch = map(torch.stack, zip(*batch))
        return batch

    def _sample_sequence(self):

        # Choose an episode using 50:50 sampling
        episode_keys = list(self.completed_episodes.keys())
        if self.recent_threshold > np.random.random():
            # Sample using a triangular distribution with min value 0, mode and max value equal to the number of stored episodes
            # favouring latest experience
            episode_key = episode_keys[int(np.floor(np.random.triangular(0, len(episode_keys), len(episode_keys), 1)))]
        else:
            episode_key = np.random.choice(episode_keys)

        episode = self.completed_episodes[episode_key]

        # Choose a length
        episode_length = self.episode_length(episode) + 1
        sequence_length = episode_length

        if self.maximum_length:
            sequence_length = min(sequence_length, self.maximum_length)

        # Randomize length to avoid having all the episodes ending at the same time
        sequence_length -= np.random.randint(self.minimum_length)
        sequence_length = max(self.minimum_length, sequence_length)

        # Get the last chosen length transitions of the episode
        upper = episode_length - sequence_length + 1

        if self.prioritize_endings:
            upper += self.minimum_length
        
        idx = min(np.random.randint(upper), episode_length - sequence_length)
        sequence =  {
            key: self.convert(value[idx: idx + sequence_length]) for key, value in episode.items()
        }

        sequence["is_first"] = np.zeros(len(sequence["action"]), bool)
        sequence["is_first"][0] = True

        if self.maximum_length:
            assert self.minimum_length <= len(sequence["action"]) <= self.maximum_length

        return sequence

    def count_episodes(self, path):
        filenames = list(glob.glob(f"{path}/*.npz"))
        num_episodes = len(filenames)
        num_steps = sum(int(str(os.path.basename(n)).split("-")[3]) - 1 for n in filenames)
        return num_episodes, num_steps

    def episode_length(self, episode):
        return len(episode["action"]) - 1
    
    def convert(self, value):
        # Convert to np types
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
        