import os
import torch
from torch.utils.data import Dataset

class MultimodalPairsDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the dataset by loading paths to all .pt files in the directory.
        """
        self.data_dir = data_dir
        self.file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'multimodal_pair' in file and file.endswith('.pt')]

    def __len__(self):
        """
        Return the number of multimodal pairs in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and return the specified multimodal pair from a .pt file.
        """
        file_path = self.file_paths[idx]
        multimodal_pair = torch.load(file_path)
        force_tensor = multimodal_pair['force']
        phrase_one_hot = multimodal_pair['phrase_one_hot']

        return force_tensor, phrase_one_hot
    
import random
from torch.utils.data import Sampler

class MultimodalPairsSampler(Sampler):
    def __init__(self, dataset, no_action_prob=0.33, num_samples_per_epoch=None):
        """
        Custom sampler to control probabilities of sampling 'no_action' pairs.
        
        Args:
            dataset: The dataset object.
            no_action_prob: Probability of sampling a 'no_action' pair.
            num_samples_per_epoch: Total number of samples per epoch.
        """
        self.dataset = dataset
        self.no_action_prob = no_action_prob

        # Separate indices into action and no_action groups
        self.action_indices = [i for i, path in enumerate(dataset.file_paths) if "no_action" not in path]
        self.no_action_indices = [i for i, path in enumerate(dataset.file_paths) if "no_action" in path]

        # Set number of samples per epoch
        self.num_samples_per_epoch = num_samples_per_epoch if num_samples_per_epoch else len(dataset)

    def __iter__(self):
        """
        Generate indices with specified probabilities for no_action pairs.
        """
        indices = []
        while len(indices) < self.num_samples_per_epoch:
            if random.random() < self.no_action_prob:
                indices.append(random.choice(self.no_action_indices))  # Sample no_action pair
            else:
                indices.append(random.choice(self.action_indices))  # Sample action pair
        random.shuffle(indices)  # Shuffle the final indices
        return iter(indices)

    def __len__(self):
        """
        Total number of samples per epoch.
        """
        return self.num_samples_per_epoch