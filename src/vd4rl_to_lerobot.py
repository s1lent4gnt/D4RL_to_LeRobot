import os
import pathlib
import random
from typing import Optional, Tuple
import collections
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

import gym
import numpy as np
import torch
from pathlib import Path

import dmcgym

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

# VD4RL Constants
VD4RL_DIR = os.path.expanduser("~/.vd4rl")

class UniversalSeed(gym.Wrapper):
    def seed(self, seed: int):
        seeds = self.env.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)
        return seeds
    
from gym.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stacking_key: str = "pixels"):
        super().__init__(env)
        self._num_stack = num_stack
        self._stacking_key = stacking_key

        assert stacking_key in self.observation_space.spaces
        pixel_obs_spaces = self.observation_space.spaces[stacking_key]

        self._env_dim = pixel_obs_spaces.shape[-1]

        low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
        new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
        self.observation_space.spaces[stacking_key] = new_pixel_obs_spaces

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._num_stack):
            self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs, reward, done, info
    
class RepeatAction(gym.Wrapper):
    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray):
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if done:
                break

        return obs, total_reward, done, combined_info
    
from gym.wrappers.pixel_observation import PixelObservationWrapper

def wrap_pixels(
    env: gym.Env,
    action_repeat: int,
    image_size: int = 84,
    num_stack: Optional[int] = 3,
    camera_id: int = 0,
    pixel_keys: Tuple[str, ...] = ("pixels",),
) -> gym.Env:
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    env = UniversalSeed(env)
    env = gym.wrappers.RescaleAction(env, -1, 1)

    env = PixelObservationWrapper(
        env,
        pixels_only=True,
        render_kwargs={
            "pixels": {
                "height": image_size,
                "width": image_size,
                "camera_id": camera_id,
            }
        },
        pixel_keys=pixel_keys,
    )

    if num_stack is not None:
        env = FrameStack(env, num_stack=num_stack)

    env = gym.wrappers.ClipAction(env)

    return env, pixel_keys

def wrap(env):
    """Wrap environment for pixel observations."""
    return wrap_pixels(
        env,
        action_repeat=2,
        image_size=64,
        num_stack=3,
        camera_id=0,
    )

def download_vd4rl_dataset(env_name: str, level: str, dataset_path: Optional[str] = None):
    """Download specific environment and level from VD4RL dataset."""
    print(f"Downloading VD4RL dataset for {env_name}-{level}...")
    
    # Create directory structure with proper path expansion
    dataset_dir = os.path.expanduser(dataset_path) if dataset_path else os.path.expanduser(VD4RL_DIR)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Make sure folder_path ends with /* to get all contents
    pattern = f"vd4rl/main/{env_name}/{level}/64px/*"
    
    # Download the folder
    local_path = snapshot_download(
        repo_id="conglu/vd4rl",
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=dataset_dir,
        ignore_patterns=["*.gitattributes", "*.gitignore", "README.md"]
    )

def get_dataset_dir(env, dataset_level, dataset_path):
    """Get the directory path for the VD4RL dataset."""
    env_name = env.unwrapped.spec.id
    env_name = "_".join(env_name.split("-")[:-1])
    
    # First try to find existing dataset
    dataset_path = dataset_path if dataset_path else VD4RL_DIR
    dataset_dir = os.path.join(dataset_path, "vd4rl/main", f"{env_name}/{dataset_level}/64px/")
    dataset_dir = pathlib.Path(os.path.expanduser(dataset_dir))
    print(f"Dataset directory: {dataset_dir}")
    # If dataset doesn't exist, download it
    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}. Downloading from HuggingFace...")
        download_vd4rl_dataset(env_name, dataset_level, dataset_path)
        print(f"Dataset downloaded to {dataset_dir}")
    
    return dataset_dir

def load_episodes(directory, capacity=None, keep_temporal_order=False):
    """Load episode data from the directory."""
    filenames = sorted(directory.glob("*.npz"))
    if not keep_temporal_order:
        print("Shuffling order of offline trajectories!")
        random.Random(0).shuffle(filenames)
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split("-")[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
                # Conversion for older versions of npz files
                if "is_terminal" not in episode:
                    episode["is_terminal"] = episode["discount"] == 0.0
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        episodes[str(filename)] = episode
    return episodes

def convert(value):
    """Convert numpy arrays to appropriate dtypes."""
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value

class VD4RLDataset:
    """VD4RL dataset handler."""
    def __init__(
        self,
        env: gym.Env,
        dataset_level: str,
        pixel_keys: tuple = ("pixels",),
        capacity: int = 500_000,
        dataset_path: Optional[str] = None,
    ):
        self.env = env
        self.dataset_level = dataset_level
        self.pixel_keys = pixel_keys
        self.capacity = capacity
        
        # Load the dataset
        dataset_dir = get_dataset_dir(env, dataset_level, dataset_path)
        print(f"Dataset directory: {dataset_dir}")
        self.episodes = load_episodes(dataset_dir, capacity=capacity, keep_temporal_order=True)
        self.current_episode = None
        self.episode_idx = 0
        self.frame_idx = 0
        
        # Get the first episode to determine shapes
        first_episode = next(iter(self.episodes.values()))
        self.observation_shape = first_episode["image"].shape[1:]
        self.action_shape = first_episode["action"].shape[1:]
        self.framestack = env.observation_space[pixel_keys[0]].shape[-1]
        
    def __len__(self):
        return sum(len(episode["action"]) for episode in self.episodes.values())
    
    def get_next_transition(self):
        """Get the next transition while ensuring episodes are not mixed."""
        if self.current_episode is None or self.frame_idx >= len(self.current_episode["action"]):
            # Move to the next episode
            episode_keys = list(self.episodes.keys())
            if self.episode_idx >= len(episode_keys):
                return None  # No more episodes available

            self.current_episode = self.episodes[episode_keys[self.episode_idx]]
            self.episode_idx += 1
            self.frame_idx = 0  # Reset frame index for new episode

        episode = self.current_episode
        frame_idx = self.frame_idx
        self.frame_idx += 1

        # Stack frames for observations
        stacked_frames = []
        next_stacked_frames = []

        for i in range(self.framestack):
            idx = max(0, frame_idx - (self.framestack - i - 1))
            stacked_frames.append(episode["image"][idx])
            next_idx = min(len(episode["image"]) - 1, idx + 1)
            next_stacked_frames.append(episode["image"][next_idx])

        transition = {
            "observations": {self.pixel_keys[0]: np.stack(stacked_frames, axis=-1)},
            "actions": episode["action"][frame_idx],
            "rewards": episode["reward"][frame_idx],
            "next_observations": {self.pixel_keys[0]: np.stack(next_stacked_frames, axis=-1)},
            "dones": bool(episode["is_terminal"][frame_idx]),  # Ensure it's a boolean
            "terminals": bool(episode["is_terminal"][frame_idx]),  # Same as `dones`
            "is_first": bool(episode["is_first"][frame_idx]),
            "is_last": bool(episode["is_last"][frame_idx]),
        }

        return transition



def build_features(observation_shape, action_shape):
    """Build features dictionary for LeRobot dataset."""
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": observation_shape,
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": action_shape,
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }
    return features

def convert_vd4rl_to_lerobot(
    env: gym.Env,
    dataset_level: str,
    repo_id: str,
    task_description: str,
    fps: int = 10,
    push_to_hub: bool = True,
    capacity: int = 500_000,
    dataset_path: Optional[str] = None,
):
    """
    Convert a VD4RL dataset to LeRobot format, ensuring episodes are stored correctly.
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # Load VD4RL dataset
    vd4rl_dataset = VD4RLDataset(
        env=env,
        dataset_level=dataset_level,
        pixel_keys=("pixels",),
        capacity=capacity,
        dataset_path=dataset_path
    )
    
    # Create LeRobot dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="dmcontrol",
        features=build_features(vd4rl_dataset.observation_shape, vd4rl_dataset.action_shape),
        image_writer_threads=4,
    )

    while True:
        transition = vd4rl_dataset.get_next_transition()
        if transition is None:
            break  # Stop when there are no more transitions

        frame = {
            "observation.image": torch.from_numpy(transition['observations']["pixels"][:, :, :, 1]),
            "action": torch.from_numpy(transition['actions']),
            "next.reward": torch.tensor([transition['rewards']], dtype=torch.float32),
            "next.done": torch.tensor([transition['dones']], dtype=torch.bool),
        }

        lerobot_dataset.add_frame(frame)

        if transition["is_last"]:
            lerobot_dataset.save_episode(task=task_description)

    # Consolidate and upload dataset
    lerobot_dataset.consolidate()
    if push_to_hub:
        lerobot_dataset.push_to_hub()

    return lerobot_dataset


if __name__ == "__main__":
    # Example usage with dataset downloading

    # Create environment
    env = gym.make("cheetah-run-v0")
    env, pixel_keys = wrap(env)  # Wrap environment for pixel observations
    
    # Convert dataset
    dataset = convert_vd4rl_to_lerobot(
        env=env,
        dataset_level="expert",
        repo_id="lilkm/vd4rl-test",
        task_description="Control a cheetah-run with expert-level demonstrations",
        fps=10,
        push_to_hub=True
    )