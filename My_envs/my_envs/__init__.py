from gymnasium.envs.registration import register

register(
    id="my_envs/RLAD-v0",
    entry_point="my_envs.envs:RLADEnv",
    max_episode_steps=100,
)
