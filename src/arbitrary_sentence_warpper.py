import numpy as np
from sentence_mapper import SentenceMapper

modifiers = [
    '',
    'slightly',
    'greatly',
    'smoothly',
    'sharply',
    'slowly',
    'quickly',
    'lightly',
    'significantly',
    'softly',
    'harshly',
    'gradually',
    'immediately',
]

directions = [
    'backward',
    'backward and down',
    'backward and left',
    'backward and right',
    'backward and up',
    'down',
    'down and forward',
    'down and left',
    'down and right',
    'forward',
    'forward and left',
    'forward and right',
    'forward and up',
    'left',
    'left and up',
    'right',
    'right and up',
    'up',
]

vocabulary = [(f'Move {modifier} {direction}.', np.array([modifier, direction.split(' ')[0], direction.split(' ')[-1] if 'and' in direction else ''], dtype='U16'))
              for modifier in modifiers for direction in directions]
vocabulary += [('', np.array(['', '', ''], dtype='U16'))]

ARBITRARY_SENTENCE_MAPPER = SentenceMapper(vocabulary)