import shutil
import minari
import torch
from pathlib import Path
import numpy as np
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
        "next.success": {
            "dtype": "bool",
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

def convert_d4rl_to_lerobot(d4rl_dataset_name: str, repo_id: str, task_description: str, fps: int = 10, push_to_hub: bool = True):
    """
    Convert a D4RL dataset to LeRobot format.
    
    Args:
        d4rl_dataset_name: Name of the D4RL dataset (e.g., "D4RL/door/human-v2")
        repo_id: HuggingFace repo ID for the converted dataset (e.g., "lilkm/D4RL_door_human-v2")
        task_description: Description of the task
        fps: Frames per second for the dataset
        push_to_hub: Whether to push the converted dataset to HuggingFace
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # Load D4RL dataset using Minari
    dataset = minari.load_dataset(d4rl_dataset_name)
    
    # Get a sample episode to determine shapes
    sample_episode = next(dataset.iterate_episodes())
    obs_shape = sample_episode.observations[0].shape
    action_shape = sample_episode.actions[0].shape
    
    # Build features dictionary
    features = build_features(obs_shape, action_shape)
    
    # Create LeRobot dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="open-door",
        features=features
    )
    
    # Convert each episode
    for episode_data in dataset.iterate_episodes():
        num_frames = len(episode_data.actions)
        
        for frame_idx in range(num_frames):
            # Prepare frame data
            frame = {
                "observation.state": torch.from_numpy(episode_data.observations[frame_idx]).float(),
                "action": torch.from_numpy(episode_data.actions[frame_idx]).float(),
                "next.reward": torch.tensor([episode_data.rewards[frame_idx]], dtype=torch.float32),
                "next.success": torch.tensor([episode_data.terminations[frame_idx]], dtype=torch.bool),
                "next.done": torch.tensor([np.logical_or(episode_data.terminations[frame_idx], episode_data.truncations[frame_idx])], dtype=torch.bool)
            }
            
            # Add frame to dataset
            lerobot_dataset.add_frame(frame)
        
        # Save episode
        lerobot_dataset.save_episode(task=task_description)
    
    # Consolidate dataset
    lerobot_dataset.consolidate()
    
    # Push to hub if requested
    if push_to_hub:
        lerobot_dataset.push_to_hub()
    
    return lerobot_dataset

if __name__ == "__main__":
    # Example usage
    d4rl_dataset_name = "D4RL/door/human-v2"
    repo_id = "lilkm/door-human-v2"  # Replace with your HuggingFace username
    task_description = "Open the door using human demonstrations"
    
    dataset = convert_d4rl_to_lerobot(
        d4rl_dataset_name=d4rl_dataset_name,
        repo_id=repo_id,
        task_description=task_description,
        fps=10,
        push_to_hub=True
    )