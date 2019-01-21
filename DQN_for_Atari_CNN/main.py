
import DQN_brian
import gym
import numpy as np
REPLAY_START = 20000

env = gym.make('Pong-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DQN_brian.Deep_Q_Network(
                            action_size=env.action_space.n,
                            feature_size=(84, 84, 3),
                              )


step = 0
game_round = 0

# it will run GAME_ROUND round game totally
while True:
    current_state = env.reset()
    while True:
        env.render()
        # return the index of the action will be executed
        action = RL.choose_action(np.resize(current_state, (84, 84, 3)))
        next_state, reward, done, info = env.step(action)
        # print(current_state.shape)
        # save the sample into memory pool
        next_state = np.resize(next_state, (84, 84, 3))
        current_state = np.resize(current_state, (84, 84, 3))
        RL.store_in_memory(current_state, action, reward, next_state)
        # when num of sample more than 200, leaning per 5 steps
        if (step > REPLAY_START) and (step % 4 == 0):
            RL.learn()
        if (step < REPLAY_START):
            print("training will start after", REPLAY_START-step)
        current_state = next_state

        # if current_round finished,break and run next episode
        if done:
            break
        step += 1
    game_round += 1
