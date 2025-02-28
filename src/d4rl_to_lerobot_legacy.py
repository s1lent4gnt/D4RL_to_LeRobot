from typing import Optional

import d4rl
import gym
import gym.wrappers
import numpy as np

import shutil
import torch
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


def build_features(observation_shape, action_shape):
    """Build features dictionary based on observation and action shapes."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": observation_shape,
            "names": None,
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


def get_d4rl_dataset(
    env,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
):
    dataset = d4rl.qlearning_dataset(gym.make(env).unwrapped)

    if clip_action:
        dataset["actions"] = np.clip(dataset["actions"], -clip_action, clip_action)

    dones_float = np.zeros_like(dataset["rewards"])

    if "kitchen" in env:
        # kitchen envs don't set the done signal correctly
        dones_float = dataset["rewards"] == 4

    else:
        # antmaze / locomotion envs
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

    # reward scale and bias
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias

    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=np.logical_or(dataset["terminals"], dones_float),
        masks=1 - dataset["terminals"].astype(np.float32),
    )

def convert_d4rl_to_lerobot(
    d4rl_dataset_name: str, 
    repo_id: str, 
    task_description: str, 
    fps: int = 10, 
    push_to_hub: bool = True,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
):
    """
    Convert a D4RL dataset to LeRobot format.
    
    Args:
        d4rl_dataset_name: Name of the D4RL dataset (e.g., "halfcheetah-medium-v2")
        repo_id: HuggingFace repo ID for the converted dataset
        task_description: Description of the task
        fps: Frames per second for the dataset
        push_to_hub: Whether to push the converted dataset to HuggingFace
        reward_scale: Scale factor for rewards
        reward_bias: Bias term added to rewards
        clip_action: Optional value to clip actions
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # Load D4RL dataset
    dataset = get_d4rl_dataset(
        d4rl_dataset_name,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        clip_action=clip_action
    )
    
    # Get shapes from the first entry
    obs_shape = dataset['observations'][0].shape
    action_shape = dataset['actions'][0].shape
    
    # Build features dictionary
    features = build_features(obs_shape, action_shape)
    
    # Create LeRobot dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="open-door",  # You might want to make this configurable
        features=features
    )
    
    # Process the data
    num_frames = len(dataset['actions'])
    episode_start = 0
    
    for frame_idx in range(num_frames):
        # Check if this is the end of an episode
        is_episode_end = dataset['dones'][frame_idx]
        
        # Prepare frame data
        frame = {
            "observation.state": torch.from_numpy(dataset['observations'][frame_idx]).float(),
            "action": torch.from_numpy(dataset['actions'][frame_idx]).float(),
            "next.reward": torch.tensor([dataset['rewards'][frame_idx]], dtype=torch.float32),
            "next.done": torch.tensor([dataset['dones'][frame_idx]], dtype=torch.bool)
        }
        
        # Add frame to dataset
        lerobot_dataset.add_frame(frame)
        
        # If this is the end of an episode, save it
        if is_episode_end:
            lerobot_dataset.save_episode(task=task_description)
            episode_start = frame_idx + 1
    
    # Save the last episode if it hasn't been saved
    if episode_start < num_frames:
        lerobot_dataset.save_episode(task=task_description)
    
    # Consolidate dataset
    lerobot_dataset.consolidate()
    
    # Push to hub if requested
    if push_to_hub:
        lerobot_dataset.push_to_hub()
    
    return lerobot_dataset


if __name__ == "__main__":
    # Example usage
    d4rl_dataset_name = "halfcheetah-medium-replay-v2"
    repo_id = "lilkm/halfcheetah-medium-replay-v2"  # Replace with your HuggingFace username
    task_description = "Open the door using human demonstrations"
    
    dataset = convert_d4rl_to_lerobot(
        d4rl_dataset_name=d4rl_dataset_name,
        repo_id=repo_id,
        task_description=task_description,
        fps=10,
        push_to_hub=True
    )