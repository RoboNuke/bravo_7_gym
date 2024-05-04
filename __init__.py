from gymnasium.envs.registration import register

register(
    id="Bravo7Base-v0",
    entry_point="bravo_7_gym.envs:Bravo7Env",
    max_episode_steps=100,
)

