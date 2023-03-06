import os
import csv
import random
from TradingEnv import Trading
import numpy as np
import tensorflow as tf

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

def train(env: Trading, qmodel_base, model_train=None, _model_target=None,
          use_accuracy=True,
          learning_rate=0.00025, random_cases=5000, epsilon=1.0,
          train_start_=0, max_steps_per_episode=1500,
          test_start_=-1, steps_test=2000):
    train_start = train_start_
    test_start = max_steps_per_episode if test_start_ == -1 else test_start_

    model = qmodel_base.create_q_model()
    if model_train:
        print("setting weights for model train")
        model.set_weights(model_train.get_weights())
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = qmodel_base.create_q_model()
    if _model_target:
        print("setting weights for model_target")
        model_target.set_weights(_model_target.get_weights())

    # Configuration parameters for the whole setup
    gamma = 0.99  # Discount factor for past rewards
    epsilon = epsilon  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 500  # Size of batch taken from replay buffer

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Number of frames to take random action and observe output
    epsilon_random_frames = random_cases
    # Number of frames for exploration
    epsilon_greedy_frames = 100000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 10
    # How often to update the target network
    update_target_network = 10000
    # Using huber loss for stability
    # loss_function = tf.keras.losses.Huber()
    loss_function = tf.keras.losses.MeanSquaredError()

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

    score_test_best = {"score": 0, "revenue": 0, "dd": 0}
    risk_test_best = {"score": 0, "revenue": 0, "dd": 0}

    while True:  # Run until solved

        state = env.reset(start_bar=train_start, steps_max=max_steps_per_episode)
        episode_reward = 0
        episode_reward_scaled = 0

        for timestep in range(0, max_steps_per_episode):
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(qmodel_base.num_actions)
            else:
                # Predict action Q-values
                # From environment state
                action_probs = model(state, training=True)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward_scaled, reward, done, _, _ = env.step(action, verbose=False)

            episode_reward += reward
            episode_reward_scaled += reward_scaled

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            if use_accuracy:
                rewards_history.append(reward_scaled)
            else:
                rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = [state_history[i] for i in indices]
                state_next_sample = [state_next_history[i] for i in indices]
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                # done_sample = tf.convert_to_tensor(
                #     [float(done_history[i]) for i in indices]
                # )
                done_sample_last = tf.convert_to_tensor(
                    [float(10 if done_history[i] == 1 else 0) for i in indices]
                )
                done_sample_bool = tf.convert_to_tensor(
                    [float(1 if done_history[i] == 1 else 0) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability

                x0 = np.reshape(
                    np.array([x[0] for x in state_next_sample]),
                    (batch_size, qmodel_base.num_non_lstm))
                x1 = np.reshape(
                    np.array([x[1] for x in state_next_sample]),
                    (batch_size, qmodel_base.history_1 * qmodel_base.vector_1)
                )
                x2 = np.reshape(
                    np.array([x[2] for x in state_next_sample]),
                    (batch_size, qmodel_base.history_2 * qmodel_base.vector_2)
                )
                future_rewards = model_target.predict([x0, x1, x2], verbose=False)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample_bool) - done_sample_last
                # Create a mask, so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, qmodel_base.num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    x0 = np.reshape(
                        np.array([x[0] for x in state_sample]),
                        (batch_size, qmodel_base.num_non_lstm)
                    )
                    x1 = np.reshape(
                        np.array([x[1] for x in state_sample]),
                        (batch_size, qmodel_base.history_1 * qmodel_base.vector_1)
                    )
                    x2 = np.reshape(
                        np.array([x[2] for x in state_sample]),
                        (batch_size, qmodel_base.history_2 * qmodel_base.vector_2)
                    )
                    q_values = model([x0, x1, x2])

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the target network with new weights
                model_target.set_weights(model.get_weights())
                tf.keras.backend.clear_session()
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done != 0:
                print(f"{env.model_name} {frame_count}. "
                      f"training score: {episode_reward:.2f}/{episode_reward_scaled}/{env.max_drawdown:.2f} ({done}), "
                      f"running_reward: {running_reward:.2f} "
                      f"best rev: {score_test_best['revenue']:.2f}/{score_test_best['dd']:.2f} "
                      f"best risk: {risk_test_best['revenue']:.2f}/{risk_test_best['dd']:.2f}"
                      )
                if done == 2:
                    test_score = 0
                    test_score_scaled = 0

                    done_test = 0
                    observation = env.reset(start_bar=test_start, steps_max=steps_test)

                    while done_test != 2:
                        action_probs = model(observation, training=False)
                        # Take best action
                        action = tf.argmax(action_probs[0]).numpy()
                        observation, reward_scaled, reward_base, done_test, _, _ = env.step(action, verbose=False)
                        test_score += reward_base
                        test_score_scaled += reward_scaled
                    print(f"{test_score:.2f} / {test_score_scaled} / {env.max_drawdown:.2f}"
                          f" / {(test_score / env.max_drawdown):.2f}")
                    if test_score >= score_test_best["score"]:
                        print("save best revenue")
                        model.save_weights(f"best_revenue_{env.model_name}")
                        score_test_best["score"] = test_score
                        score_test_best["revenue"] = test_score
                        score_test_best["dd"] = env.max_drawdown
                    if test_score / env.max_drawdown >= risk_test_best["score"]:
                        print("save best risk")
                        model.save_weights(f"best_risk_{env.model_name}")
                        risk_test_best["score"] = test_score / env.max_drawdown
                        risk_test_best["revenue"] = test_score
                        risk_test_best["dd"] = env.max_drawdown
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if (score_test_best["score"] > 2 or risk_test_best["score"] > 9) and running_reward > 10:
            # or (score_test_best > 1.5 and score_train_best > 5):  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break

    return model


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

    for timestep in range(0, max_steps_per_episode):
        action_probs = model_backtest(observation, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        observation, _, reward_base, done_test, bar_start, bar_end = env.step(action, verbose=False)

        train_score += reward_base

        trades_history.append([bar_start, bar_end, action, reward_base, train_score,
                               action_probs[0][0].numpy(),
                               action_probs[0][1].numpy(),
                               "train"]
                              )

    test_score = 0
    done_test = 0
    observation = env.reset(start_bar=test_start, steps_max=steps_test)

    while done_test != 2:
        action_probs = model_backtest(observation, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        observation, _, reward_base, done_test, bar_start, bar_end = env.step(action, verbose=False)
        test_score += reward_base

        trades_history.append([bar_start, bar_end, action, reward_base, test_score,
                               action_probs[0][0].numpy(),
                               action_probs[0][1].numpy(),
                               "test"]
                              )

    print(f"train score: {train_score:.2f} | test score: {test_score:.2f} "
          f"| dd: {env.max_drawdown:.2f} | test risk: {test_score / env.max_drawdown:.2f}")

    with open("ai_trades.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(trades_history)
