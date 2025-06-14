import numpy as np
from numpy.typing import NDArray
import os
from typing import Tuple
import pickle
import scipy.interpolate
from globals import FORCE_SAMPLE_COUNT, FORCE_CHANNEL_COUNT, PHRASE_WORD_COUNT, FORCE_CURVE_DURATION


class Dataset:
    def __init__(self, directory: str):
        self.directory = directory

    def load(self, exclude_second_phase: bool = True) -> Tuple[NDArray, NDArray]:
        trials = [pickle.load(open(os.path.join(self.directory, file), 'rb'))
                  for file in os.listdir(self.directory) if file.endswith('.pkl')]
        if exclude_second_phase:
            trials = [trial for trial in trials if trial['user_id'] < 10]

        user_count = max([trial['user_id'] for trial in trials]) + 1
        trials_per_user = max([trial['trial_number'] for trial in trials]) + 1
        max_word_length = max([max(len(trial['adverb']), len(
            trial['first_cartesian_direction'])) for trial in trials])

        force_data = np.zeros((user_count, trials_per_user,
                               FORCE_SAMPLE_COUNT, FORCE_CHANNEL_COUNT))
        phrase_data = np.zeros(
            (user_count, trials_per_user, PHRASE_WORD_COUNT), dtype=f'U{max_word_length}')

        for trial in trials:
            time = trial['force'][0]
            force = trial['force'][1:].T

            force = force[time < FORCE_CURVE_DURATION]
            time = time[time < FORCE_CURVE_DURATION]

            force[-1] = np.zeros_like(force[-1])

            time = np.concatenate((time, [FORCE_CURVE_DURATION]))
            force = np.concatenate(
                (force, np.zeros((1, FORCE_CHANNEL_COUNT))), axis=0)

            resampled_time = np.linspace(time[0], time[-1], FORCE_SAMPLE_COUNT)
            force_interp = scipy.interpolate.interp1d(time, force, axis=0)
            resampled_force = force_interp(resampled_time)

            phrase = np.array([trial['adverb'], trial['first_cartesian_direction'],
                               trial['second_cartesian_direction']], dtype=f'U{max_word_length}')

            force_data[trial['user_id'],
                       trial['trial_number']] = resampled_force
            phrase_data[trial['user_id'], trial['trial_number']] = phrase

        force_data = force_data.reshape(-1,
                                        force_data.shape[-2], force_data.shape[-1])
        phrase_data = phrase_data.reshape(-1, phrase_data.shape[-1])

        force_data = force_data.cumsum(
            axis=1) * FORCE_CURVE_DURATION / FORCE_SAMPLE_COUNT
        phrase_data[:, 1:] = np.sort(phrase_data[:, 1:], axis=1)

        return force_data, phrase_data

    def merge_directions(self, phrase_data: NDArray) -> NDArray:
        merged_directions = np.char.add(np.char.add(
            phrase_data[:, 1], ' '), phrase_data[:, 2])
        merged_directions = np.char.lstrip(merged_directions)
        phrase_data = np.stack((phrase_data[:, 0], merged_directions), axis=-1)
        return phrase_data

    def merge_phrase(self, phrase_data: NDArray) -> NDArray:
        phrase = phrase_data[:, 1]
        phrase = np.char.add(np.char.add(phrase, ' '), phrase_data[:, 2])
        phrase = np.char.add(np.char.strip(phrase), '.')
        phrase = np.char.add(np.char.add(phrase_data[:, 0], ' '), phrase)
        phrase = np.char.add('Move ', phrase)
        return phrase
