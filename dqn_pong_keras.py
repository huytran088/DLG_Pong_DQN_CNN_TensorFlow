import numpy as np
import gym
#for neural network model
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
#for plotting
import matplotlib.pyplot as plt
from datetime import datetime
import os

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 6400 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    return processed_observation

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  # we go from last reward to first one so we don't have to do exponentiations
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

def create_model(lr):
    model = Sequential()
    # hidden layer takes a pre-processed frame as input, and has 200 units. Simple layer architectur of 200 x1, 1x1
    model.add(Dense(
        units=200,
        input_dim=80 * 80,
        activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform()) # Glorot is a form of Xavier Initalization https://keras.io/api/layers/initializers/#glorot_uniform
    )
    # output layer — we use a Sigmoid here, in order to get a 0, or 1 value to represent ACTION UP
    model.add(Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.GlorotUniform())
    )
    # compile the model using traditional Machine Learning losses and optimizers
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer =opt,
                  metrics = ['accuracy'])
    return model
def play():
    env = gym.make("Pong-v0")
    resume = False #change it to true if you already trained the agent previously
    train_episodes = 700

    # hyperparameters
    epoch_update_weight = 1500 # tthe number of training samples to work through before the model’s weights are updated
    epochs_number = 3 # the number times that the learning algorithm will work through the entire training dataset. 
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # the exploitation rate of the agent
    learning_rate = 1e-4 # pass in create_model(lr) to set the learning_rate for the optimizer
    input_dimensions = 80*80

    # game parameters
    episode_observations, episode_actions, episode_rewards = [], [], []
    reward_sum = 0
    episode_number = 0
    running_reward = None


    #initialize DQN model
    input_dimensions = 80*80
    model = create_model(learning_rate)
    if resume:
        path = os.path.join('pong_model_checkpoint.h5')
        model.load_weights(path)
    epochs_before_saving = 100 # use for saving model weight every 100 episodes

    #Plotting
    loss_buffer = []
    reward_buffer = []
    total_episodes_buffer = []


    observation = env.reset() # This gets us the image
    prev_input = None
    # main training loop
    #while (True):
    while episode_number < train_episodes:
        env.render() # comment this line if you dont want to see the agent play Pong in real time
        cur_observations = preprocess_observations(observation)
        processed_observations = cur_observations - prev_input if prev_input is not None else np.zeros(input_dimensions)
        prev_input = cur_observations

        # forward the policy network and sample action according to the probability distribution
        up_probability = model.predict(np.expand_dims(processed_observations, axis=1).T)

        action = choose_action(up_probability)
        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0

        # log the input and label to train later
        episode_observations.append(processed_observations)
        episode_actions.append(fake_label)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        reward_sum += reward

        if done:
            total_episodes_buffer.append(episode_number)

            # discount the rewards based on the actions taken
            episode_action_reward_discounted = discount_rewards(episode_rewards, gamma)

            # training - meaning that performing backprop to update weights as well as running Gradient Descent
            # as the fit() in keras is called, it runs the gradient descent optimizer algorithm as defined earlier
            hist = model.fit(x=np.vstack(episode_observations),
                             y=np.vstack(episode_actions),
                             batch_size =epoch_update_weight,
                             verbose=1,
                             epochs = epochs_number,
                             sample_weight=episode_action_reward_discounted
                             )
            
            loss_buffer.append(hist.history['loss'])
            # Saving the weights used by our model
            if episode_number % epochs_before_saving == 0:
                if os.path.exists('pong_model_checkpoint.h5'):
                    os.remove('pong_model_checkpoint.h5')
                model.save_weights('pong_model_checkpoint.h5')

            observation = env.reset()  # reset env
            # simutaniously let the angent to explore the environment at .01 rate and exploit the environment at .99 rate
            running_reward = reward_sum if running_reward is None else running_reward * decay_rate + reward_sum *(1-decay_rate)
            reward_buffer.append(running_reward)

            print('resetting env. episode %f. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
            # Reinitialization
            episode_observations, episode_actions, episode_rewards = [], [], []
            reward_sum = 0
            prev_input = None
            episode_number += 1
    env.close() #if you run this on your local machine you need to close the env at the end
    plt.figure(1)
    plt.plot(total_episodes_buffer,loss_buffer)
    plt.title('Model Loss over The number of Episodes')
    plt.ylabel('Model Loss')
    plt.xlabel('Episodes')

    plt.figure(2)
    plt.plot(total_episodes_buffer,reward_buffer)
    plt.title('Rewards Earned over The number of Episodes')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.show()        
play()
