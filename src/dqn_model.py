import keras
import numpy as np
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

INPUT_SHAPE = (64, 64)


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        # Arguments
            action (int): Action given to the environment
        # Returns
            Processed action given to the environment
        Actions seem to map to
        0,1 nothing
        2,4 up
        3,5 down
        """
        # nothing
        if action == 0:
            return 0
        # up
        elif action == 2:
            return 2
        # Down
        elif action == 1:
            return 3


def cnn_model():
    """
    Loads convolutional neural network
    :return: keras model
    """
    # keras model
    action_space = np.zeros(3)
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(8,) + INPUT_SHAPE))
    # make channels first
    model.add(keras.layers.Permute((2, 3, 1)))
    model.add(keras.layers.Convolution2D(16, (8, 8), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Convolution2D(32, (4, 4), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Convolution2D(64, (2, 2), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(3, activation='linear'))
    return model


def dqn_model(model):
    """
    Create the deep Q-learning agent.
    :param model: keras model
    :return: dqn model
    """
    # RL model
    # define memory
    memory = SequentialMemory(limit=10000, window_length=8)
    # search policy with anneling schedual
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=100000)
    processor = AtariProcessor()
    # get number of actions
    nb_actions = 3  # env.action_space.n
    # create agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=10000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile('Adam', metrics=['mae'])
    return dqn

def load_saved_model(weight_path):
    """
    Returns saved model
    :param weight_path:
    :return:
    """
    model = cnn_model()
    dqn = dqn_model(model)
    dqn.load_weights(weight_path)
    return dqn
