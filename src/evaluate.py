agent.load('models/flappy_bird_model.h5')

state = env.reset()
state = preprocess_state(state)
done = False
while not done:
    action = agent.act(state, test=True)  # Im Testmodus ohne Exploration
    next_state, reward, done, _ = env.step(action)
    state = preprocess_state(next_state)
    env.render()
