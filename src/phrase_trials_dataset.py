from torch.utils.data import Dataset
from pathlib import Path
from phrase_trial_data import PhraseTrialData

def get_user_trial_numbers(file_name):
    # [user_id]__[trial_number]__[phrase].pkl
    name = file_name.split('.pkl')[0]
    user_id, trial_number = map(int, name.split('__')[:2])
    return user_id, trial_number

class PhraseForceTrialDataset(Dataset):
    def __init__(self, dir_path, transformation=None):
        dir = Path(dir_path)
        files = sorted([f for f in dir.iterdir() if f.is_file()], key=lambda f: get_user_trial_numbers(f.name))

        self.data = []

        for file in files:
            user_id, trial_number = get_user_trial_numbers(file.name)
            while len(self.data) < user_id:
                self.data.append([])

            data = PhraseTrialData.load_as_pandas_dataframe(file, transformation)
            self.data[user_id-1].append(data)

            self.flat_data = []
            for user_trials in self.data:
                for trial in user_trials:
                    self.flat_data.append(trial)

    def __len__(self):
        return sum(len(individual_user_data) for individual_user_data in self.data)
    
    def num_of_users(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            user_id, trial_number = index
            
            if isinstance(user_id, slice):
                user_ids = range(1, len(self.data) + 1)[user_id]
            else:
                user_ids = [user_id]

            results = []

            for uid in user_ids:
                if uid < 1 or uid > len(self.data):
                    raise IndexError(f"User ID {uid} out of range.")

                if isinstance(trial_number, slice):
                    trial_numbers = range(1, len(self.data[uid - 1]) + 1)[trial_number]
                else:
                    trial_numbers = [trial_number]

                for tid in trial_numbers:
                    if tid < 1 or tid > len(self.data[uid - 1]):
                        raise IndexError(f"Trial number {tid} out of range for User {uid}.")
                    results.append(self.data[uid - 1][tid - 1])

            return results if len(results) > 1 else results[0]

        return self.flat_data[index]