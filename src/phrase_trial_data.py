from __future__ import annotations
import os
import pickle
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import pandas as pd

class PhraseTrialData:
    @classmethod
    def load(cls, file_path: str, transformation: NDArray | None = None):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            phrase_trial_data = cls(None, None, None, None, None, None,)
            phrase_trial_data.__dict__ = data.__dict__

            if transformation is not None:
                phrase_trial_data.time = np.array(phrase_trial_data.time)
                phrase_trial_data.dt = np.array(phrase_trial_data.dt)
                phrase_trial_data.position = np.array(phrase_trial_data.position) @ transformation.T
                phrase_trial_data.velocity = np.array(phrase_trial_data.velocity) @ transformation.T
                phrase_trial_data.external_force = np.array(phrase_trial_data.external_force) @ transformation.T
                if not isinstance(phrase_trial_data.internal_force[0], int):
                    phrase_trial_data.internal_force = np.array(phrase_trial_data.internal_force) @ transformation.T

            phrase_trial_data.dt = phrase_trial_data.dt[1:]  # Shift dt
            phrase_trial_data.time = phrase_trial_data.time[:-1]  # Truncate others
            phrase_trial_data.position = phrase_trial_data.position[:-1]
            phrase_trial_data.velocity = phrase_trial_data.velocity[:-1]
            phrase_trial_data.external_force = phrase_trial_data.external_force[:-1]
            phrase_trial_data.internal_force = phrase_trial_data.internal_force[:-1]

            return phrase_trial_data
        
    @classmethod
    def load_as_pandas_dataframe(cls, file_path: str, transformation: NDArray | None = None) -> pd.DataFrame:
        phrase_trial_data = cls.load(file_path, transformation)

        data = {
            'user_id': [phrase_trial_data.user_id] * len(phrase_trial_data.time),
            'trial_number': [phrase_trial_data.trial_number] * len(phrase_trial_data.time),
            'phrase': [phrase_trial_data.phrase] * len(phrase_trial_data.time),
            'adverb': [phrase_trial_data.adverb] * len(phrase_trial_data.time),
            'first_cartesian_direction': [phrase_trial_data.first_cartesian_direction] * len(phrase_trial_data.time),
            'second_cartesian_direction': [phrase_trial_data.second_cartesian_direction] * len(phrase_trial_data.time),
            'time': phrase_trial_data.time,
            'dt': phrase_trial_data.dt
        }

        position = np.array(phrase_trial_data.position)
        velocity = np.array(phrase_trial_data.velocity)
        external_force = np.array(phrase_trial_data.external_force)
        internal_force = np.array(phrase_trial_data.internal_force if not isinstance(phrase_trial_data.internal_force[0], int) else [np.zeros(3) for _ in phrase_trial_data.internal_force])

        for prefix, array in zip(
            ['position', 'velocity', 'external_force', 'internal_force'],
            [position, velocity, external_force, internal_force]
        ):
            for i in range(array.shape[1]):
                data[f'{prefix}_{i}'] = array[:, i]

        return pd.DataFrame(data)

    def __init__(self, user_id: int, trial_number: int, phrase: str, adverb: str, first_cartesian_direction: str, second_cartesian_direction: str):
        self.user_id = user_id
        self.trial_number = trial_number
        self.phrase = phrase
        self.adverb = adverb
        self.first_cartesian_direction = first_cartesian_direction
        self.second_cartesian_direction = second_cartesian_direction
        self.time = []
        self.dt = [] # this should be shifted back by 1 because dt is measured from t-1 to t but that would implied the dt recorded at t is really the dt starting at t-1
        self.position = []
        self.velocity = []
        self.external_force = []
        self.internal_force = []

    def append(self, time: float, dt: float, position: NDArray, velocity: NDArray, external_force: NDArray, internal_force: NDArray) -> None:
        self.time.append(time)
        self.dt.append(dt)
        self.position.append(position)
        self.velocity.append(velocity)
        self.external_force.append(external_force)
        self.internal_force.append(internal_force)

    def save(self, index: int, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)

        safe_phrase = self.phrase.replace(" ", "_")

        file_name = f"{self.user_id}__{index}__{safe_phrase}.pkl"

        file_path = os.path.join(dir, file_name)

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

### COPY PASTE BELOW INTO MAIN MODULE

# import time

# class Timer:
#     def __init__(self) -> None:
#         self.reset()

#     def dt(self) -> float:
#         t = self.t()
#         dt = t - self.last_t
#         self.last_t = t
#         return dt

#     def t(self) -> float:
#         return time.time() - self.start_time

#     def reset(self) -> None:
#         self.start_time = time.time()
#         self.last_t = 0.0


# class ForceCurve:
#     def __init__(self, direction, duration, peak_force, ramp_up_pct=0.5, hold_pct=0.0, ramp_down_pct=0.5):
#         self.direction = direction
#         self.ramp_up_time = ramp_up_pct * duration
#         self.hold_time = hold_pct * duration
#         self.ramp_down_time = ramp_down_pct * duration
#         self.duration = duration
#         self.peak_force = peak_force
#         self.start_time = 0.0
#         self.stop_time = 0.0
    
#     def start(self):
#         self.start_time = time.time()
    
#     def stop(self):
#         self.stop_time = time.time()
    
#     def get_force(self):
#         t = time.time() - self.start_time
#         if t <= self.ramp_up_time:
#             magnitude = self.peak_force * t / self.ramp_up_time
#         elif t <= self.hold_time:
#             magnitude = self.peak_force
#         elif t <= self.ramp_down_time:
#             magnitude = self.peak_force * (1.0 - (t - self.ramp_up_time - self.hold_time) / self.ramp_down_time)
#         else:
#             magnitude = 0.0
        
#         is_done = t > self.duration
#         if is_done:
#             self.stop()

#         return self.direction * magnitude, is_done
