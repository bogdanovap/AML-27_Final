import os
import csv
import random

import numpy as np
import tensorflow as tf

from datetime import datetime

from TradingEnv import Trading

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# old_state = random.randint(0, 1000000)
old_state = 736870
random.seed(old_state)
print("Seed was:", old_state)

tf.keras.utils.set_random_seed(old_state)


#
# Implement the Deep Q-Network
#
class QModel:
    num_actions = 3

    dropout = 0.1

    num_non_lstm = 1
    history_1 = 5
    vector_1 = 8
    history_2 = 5
    vector_2 = 8
    inputs_length = num_non_lstm + history_1 * vector_1 + history_2 * vector_2
    lstm_dims = 100
    fc1_dims = 100  # 50
    fc2_dims = 100  # 25
    fc3_dims = 100  # 10

    l2_lambda = 0.0001

    initializer = tf.keras.initializers.HeNormal()

    def __init__(self, _history_1=5, _vector_1=8, _history_2=5, _vector_2=8):
        self.history_1 = _history_1
        self.history_2 = _history_2
        self.vector_1 = _vector_1
        self.vector_2 = _vector_2

    def create_q_model(self):
        # inputs = layers.Input(shape=(history, vector,))
        inputs_current = tf.keras.layers.Input(shape=(self.num_non_lstm,))

        inputs_lstm_long = tf.keras.layers.Input(shape=(self.history_1 * self.vector_1,))
        reshape_long = tf.keras.layers.Reshape((self.history_1, self.vector_1))(inputs_lstm_long)
        lstm1_long = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(reshape_long)
        lstm1_long_dropout = tf.keras.layers.Dropout(self.dropout)(lstm1_long)
        # lstm2_long = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(lstm1_long_dropout)
        # lstm2_long_dropout = tf.keras.layers.Dropout(self.dropout)(lstm2_long)
        # lstm3_long = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(lstm2_long_dropout)
        # lstm3_long_dropout = tf.keras.layers.Dropout(self.dropout)(lstm3_long)
        # lstm4_long = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(lstm3_long_dropout)
        # lstm4_long_dropout = tf.keras.layers.Dropout(self.dropout)(lstm4_long)
        lstm5_long = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=False)(lstm1_long_dropout)
        lstm5_long_dropout = tf.keras.layers.Dropout(self.dropout)(lstm5_long)

        inputs_lstm_short = tf.keras.layers.Input(shape=(self.history_2 * self.vector_2,))
        reshape_short = tf.keras.layers.Reshape((self.history_2, self.vector_2))(inputs_lstm_short)
        lstm1_short = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(reshape_short)
        lstm1_short_dropout = tf.keras.layers.Dropout(self.dropout)(lstm1_short)
        # lstm2_short = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(lstm1_short_dropout)
        # lstm2_short_dropout = tf.keras.layers.Dropout(self.dropout)(lstm2_short)
        # lstm3_short = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(lstm2_short_dropout)
        # lstm3_short_dropout = tf.keras.layers.Dropout(self.dropout)(lstm3_short)
        # lstm4_short = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=True)(lstm3_short_dropout)
        # lstm4_short_dropout = tf.keras.layers.Dropout(self.dropout)(lstm4_short)
        lstm5_short = tf.keras.layers.LSTM(self.lstm_dims, return_sequences=False)(lstm1_short_dropout)
        lstm5_short_dropout = tf.keras.layers.Dropout(self.dropout)(lstm5_short)

        concat = tf.keras.layers.concatenate([inputs_current, lstm5_long_dropout, lstm5_short_dropout])

        dense1 = tf.keras.layers.Dense(self.fc1_dims, activation="relu",
                                       kernel_initializer=self.initializer,
                                       bias_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
                                       activity_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
                                       )(concat)
        dropout1 = tf.keras.layers.Dropout(self.dropout)(dense1)
        dense2 = tf.keras.layers.Dense(self.fc2_dims, activation="relu",
                                       kernel_initializer=self.initializer,
                                       bias_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
                                       activity_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
                                       )(dropout1)
        dropout2 = tf.keras.layers.Dropout(self.dropout)(dense2)
        dense3 = tf.keras.layers.Dense(self.fc3_dims, activation="relu",
                                       kernel_initializer=self.initializer,
                                       bias_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
                                       activity_regularizer=tf.keras.regularizers.l2(self.l2_lambda),
                                       )(dropout2)
        dropout3 = tf.keras.layers.Dropout(self.dropout)(dense3)
        # prediction = tf.keras.layers.Dense(num_actions, activation="tanh")(dropout0)
        # prediction = tf.keras.layers.Dense(self.num_actions, activation="linear")(dense3)
        prediction = tf.keras.layers.Dense(self.num_actions, activation="softmax")(dropout3)

        return tf.keras.Model(inputs=[inputs_current, inputs_lstm_long, inputs_lstm_short], outputs=prediction)


# The first model makes the predictions for Q-values which are used to
# make an action.
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.

#
# Train
#
class QModelTrainer:
    qmodel_base = QModel()
    env = Trading()

    # Configuration parameters for the whole setup
    # Discount factor for past rewards
    gamma = 0.99
    # Epsilon greedy parameter
    epsilon = 1
    # Minimum epsilon greedy parameter
    epsilon_min = 0.1
    # Maximum epsilon greedy parameter
    epsilon_max = 1.0
    # Rate at which to reduce chance of random action being taken
    epsilon_interval = (epsilon_max - epsilon_min)
    # Size of batch taken from replay buffer
    batch_size = 500
    # Number of frames to take random action and observe output
    epsilon_random_frames = 5000
    # Number of frames for exploration
    epsilon_greedy_frames = 100000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 10
    # How often to update the target network
    update_target_network = 10000

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = tf.keras.optimizers.Adam()
    # Using huber loss for stability
    # loss_function = tf.keras.losses.Huber()
    loss_function = tf.keras.losses.MeanSquaredError()

    # training cycle settings
    train_start = 0
    max_steps_per_episode = 1500
    test_start = 0
    steps_test = 500
    use_accuracy = False
    model = QModel().create_q_model()
    model_target = QModel().create_q_model()

    # training cycle buffers
    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0

    # dicts to keep track of best results
    score_test_best = {"score": 0, "revenue": 0, "dd": 0}
    risk_test_best = {"score": 0, "revenue": 0, "dd": 0}

    def __init__(self,
                 env: Trading,
                 qmodel_base, model_train=None, _model_target=None,
                 use_accuracy=True,
                 learning_rate=0.00025,
                 random_cases=5000, epsilon=1.0,
                 train_start_=0, max_steps_per_episode=1500,
                 test_start_=-1, steps_test=2000
                 ):
        self.qmodel_base = qmodel_base
        self.env = env

        self.train_start = train_start_
        self.max_steps_per_episode = max_steps_per_episode
        self.test_start = max_steps_per_episode if test_start_ == -1 else test_start_
        self.steps_test = steps_test

        self.model = qmodel_base.create_q_model()
        if model_train:
            print("setting weights for model train")
            self.model.set_weights(model_train.get_weights())

        self.model_target = qmodel_base.create_q_model()
        if _model_target:
            print("setting weights for model_target")
            self.model_target.set_weights(_model_target.get_weights())

        self.use_accuracy = use_accuracy

        self.epsilon = epsilon

        self.epsilon_random_frames = random_cases

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    def prepare_states(self, states):
        x0 = np.reshape(
            np.array([x[0] for x in states]),
            (self.batch_size, self.qmodel_base.num_non_lstm))
        x1 = np.reshape(
            np.array([x[1] for x in states]),
            (self.batch_size, self.qmodel_base.history_1 * self.qmodel_base.vector_1)
        )
        x2 = np.reshape(
            np.array([x[2] for x in states]),
            (self.batch_size, self.qmodel_base.history_2 * self.qmodel_base.vector_2)
        )
        return [x0, x1, x2]

    def update_train_model(self):
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

        # Using list comprehension to sample from replay buffer
        state_sample = [self.state_history[i] for i in indices]
        state_next_sample = [self.state_next_history[i] for i in indices]
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]

        done_sample_last = tf.convert_to_tensor(
            [float(10 if self.done_history[i] == 1 else 0) for i in indices]
        )
        done_sample_bool = tf.convert_to_tensor(
            [float(1 if self.done_history[i] == 1 else 0) for i in indices]
        )

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        states_next = self.prepare_states(state_next_sample)
        future_rewards = self.model_target.predict(states_next, verbose=False)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample_bool) - done_sample_last
        # Create a mask, so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.qmodel_base.num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            states_current = self.prepare_states(state_sample)
            q_values = self.model(states_current)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def test_run(self):
        test_score = 0
        test_score_scaled = 0

        done_test = 0
        observation = self.env.reset(start_bar=self.test_start, steps_max=self.steps_test)

        # we make test run for all possible cases
        # we wait for done_test==2
        while done_test != 2:
            action_probs = self.model(observation, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            observation, reward_scaled, reward_base, _, done_test, _, _ = self.env.step(action, verbose=False)
            test_score += reward_base
            test_score_scaled += reward_scaled

        self.log_test(test_score, test_score_scaled)

        # check if we can update model if better results are acchieved
        # for revenue
        if test_score >= self.score_test_best["score"]:
            print("save best revenue")
            self.model.save_weights(f"best_revenue_{self.env.model_name}")
            self.score_test_best["score"] = test_score
            self.score_test_best["revenue"] = test_score
            self.score_test_best["dd"] = self.env.max_drawdown
        # for risk factor
        if test_score / self.env.max_drawdown >= self.risk_test_best["score"]:
            print("save best risk")
            self.model.save_weights(f"best_risk_{self.env.model_name}")
            self.risk_test_best["score"] = test_score / self.env.max_drawdown
            self.risk_test_best["revenue"] = test_score
            self.risk_test_best["dd"] = self.env.max_drawdown

    def train_episode(self):
        state = self.env.reset(start_bar=self.train_start, steps_max=self.max_steps_per_episode)
        episode_reward = 0
        episode_reward_scaled = 0

        for timestep in range(0, self.max_steps_per_episode):
            self.frame_count += 1

            # Use epsilon-greedy for exploration
            if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(self.qmodel_base.num_actions)
            else:
                # Predict action Q-values
                # From environment state
                action_probs = self.model(state, training=True)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward_scaled, reward, _, done, _, _ = self.env.step(action, verbose=False)

            episode_reward += reward
            episode_reward_scaled += reward_scaled

            # Save actions and states in replay buffer
            self.action_history.append(action)
            self.state_history.append(state)
            self.state_next_history.append(state_next)
            self.done_history.append(done)
            if self.use_accuracy:
                self.rewards_history.append(reward_scaled)
            else:
                self.rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:
                self.update_train_model()

            # update the target network with new weights
            # and clear session to free memory
            if self.frame_count % self.update_target_network == 0:
                self.model_target.set_weights(self.model.get_weights())
                tf.keras.backend.clear_session()

            # Limit the state and reward history
            if len(self.rewards_history) > self.max_memory_length:
                del self.rewards_history[:1]
                del self.state_history[:1]
                del self.state_next_history[:1]
                del self.action_history[:1]
                del self.done_history[:1]

            if done != 0:
                # log episode details
                self.log_episode(episode_reward=episode_reward, episode_reward_scaled=episode_reward_scaled, done=done)

                # if finish - make test run
                if done == 2:
                    self.test_run()

                # if finish by drawdown - reset episode
                # if is rudement, but we keep it for readability
                if done == 1:
                    # breaks current "for" loop
                    break

        if self.use_accuracy:
            return episode_reward_scaled

        return episode_reward

    def train(self):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Start training")
        while True:  # Run until solved
            self.episode_count += 1

            # run single training episode
            episode_reward = self.train_episode()

            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            running_reward = np.mean(self.episode_reward_history)

            # check conditions for finishing training
            if (self.score_test_best["score"] > 2 or self.risk_test_best["score"] > 9) and running_reward > 10:
                # or (score_test_best > 1.5 and score_train_best > 5):  # Condition to consider the task solved
                print("Solved at episode {}!".format(self.episode_count))
                break

    def log_episode(self, episode_reward=0, episode_reward_scaled=0, done=0):
        print(f"{self.env.model_name} {self.frame_count}. "
              f"training score: {episode_reward:.2f}/{episode_reward_scaled}/{self.env.max_drawdown:.2f} ({done}), "
              f"running_reward: {self.running_reward:.2f} "
              f"best rev: {self.score_test_best['revenue']:.2f}/{self.score_test_best['dd']:.2f} "
              f"best risk: {self.risk_test_best['revenue']:.2f}/{self.risk_test_best['dd']:.2f}"
              )

    def log_test(self, test_score, test_score_scaled):
        print(f"{test_score:.2f} / {test_score_scaled} / {self.env.max_drawdown:.2f}"
              f" / {(test_score / self.env.max_drawdown):.2f}")

    pass


# save trades made by model_best
def load_best_model_from_other_model(model_to, model_from):
    model_to.set_weights(model_from.get_weights())
    return model_to


def load_model_from_file(model_to_update, filename: str):
    model_to_update.load_weights(filename)
    return model_to_update


def save_model_to_file(model_to_save, filename: str):
    model_to_save.save_weights(filename)
    return model_to_save


def backtest(env: Trading, model_backtest,
             train_start_=0, max_steps_per_episode=1500,
             test_start_=-1, steps_test=2000):
    train_start = train_start_
    test_start = max_steps_per_episode if test_start_ == -1 else test_start_

    trades_history = []
    observation = env.reset(start_bar=train_start, steps_max=max_steps_per_episode)
    train_score = 0
    train_score_fact = 0

    for timestep in range(0, max_steps_per_episode):
        action_probs = model_backtest(observation, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        observation, _, reward_base, reward_fact, done_test, bar_start, bar_end = env.step(action, verbose=False)

        train_score += reward_base
        train_score_fact += reward_fact

        trades_history.append([bar_start, bar_end, action,
                               reward_base, train_score,
                               reward_fact, train_score_fact,
                               action_probs[0][0].numpy(),
                               action_probs[0][1].numpy(),
                               "train"]
                              )

    test_score = 0
    test_score_fact = 0
    done_test = 0
    observation = env.reset(start_bar=test_start, steps_max=steps_test)

    while done_test != 2:
        action_probs = model_backtest(observation, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        observation, _, reward_base, reward_fact, done_test, bar_start, bar_end = env.step(action, verbose=False)
        test_score += reward_base
        test_score_fact += reward_fact

        trades_history.append([bar_start, bar_end, action,
                               reward_base, test_score,
                               reward_fact, test_score_fact,
                               action_probs[0][0].numpy(),
                               action_probs[0][1].numpy(),
                               "test"]
                              )

    print(f"train score: {train_score:.2f} | test score: {test_score:.2f} "
          f"| dd: {env.max_drawdown:.2f} | test risk: {test_score / env.max_drawdown:.2f}")

    with open("ai_trades.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(trades_history)
