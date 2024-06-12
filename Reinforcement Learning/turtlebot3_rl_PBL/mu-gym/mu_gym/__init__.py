from gym.envs.registration import register


register(
    id='turtlebot3-ros1-dis-v0',
    entry_point='mu_gym.envs:Turtlebot3Ros1DisEnv',
)

register(
    id='turtlebot3-ros1-dis-safe-v0',
    entry_point='mu_gym.envs:Turtlebot3Ros1DisSafeEnv',
)

register(
    id='turtlebot3-ros1-dis-PBL',
    entry_point='mu_gym.envs:Turtlebot3Ros1DisPBLEnv',
)



