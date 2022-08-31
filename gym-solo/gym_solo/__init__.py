import logging

from gym.envs.registration import register
register(
    id='Solo12-v1', 
    entry_point='gym_solo.envs:SoloEnv'
)