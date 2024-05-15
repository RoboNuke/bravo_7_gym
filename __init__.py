from gymnasium.envs.registration import register

register(
    id="Bravo7Base-v0",
    entry_point="bravo_7_gym.envs:Bravo7Env",
    max_episode_steps=100,
)

register(
    id="Bravo7_DenseRepo-v0",
    entry_point="bravo_7_gym.envs.repo_dense:Bravo7RepoDense",
    max_episode_steps=100,
)

register(
    id="Bravo7FixedPegInsert-v0",
    entry_point="bravo_7_gym.envs.fixed_peg_insert:Bravo7FixedPegInsert",
    max_episode_steps=100,
)
