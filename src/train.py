from src.environment import FlappyBirdEnv
from src.agent import DQNAgent

env = FlappyBirdEnv()
agent = DQNAgent(state_size, action_size)

for e in range(episodes):
    state = env.reset()
    state = preprocess_state(state)
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay()
    # Optional: Modell speichern, Metriken aufzeichnen
