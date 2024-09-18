import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Section 1: Pre-process the Dataset
# ----------------------------------

# Load dataset
data = pd.read_csv('D:/DRL/energydata_complete.csv')

# Handle missing values by filling them with the mean of each column
data.fillna(data.mean(), inplace=True)

# Normalize features using StandardScaler
scaler = StandardScaler()
features = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9', 'Visibility', 'Tdewpoint', 'Press_mm_hg', 'Windspeed']
data[features] = scaler.fit_transform(data[features])

# Split dataset into training and testing sets
X = data[features]
y = data['Appliances']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 2: Design the Actor-Critic Algorithm using TensorFlow
# ------------------------------------------------------------

# Create the Actor Model
def create_actor():
    model = tf.keras.Sequential([
        layers.Input(shape=(21,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 actions: decrease, maintain, increase
    ])
    return model

# Create the Critic Model
def create_critic():
    model = tf.keras.Sequential([
        layers.Input(shape=(21,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Output is a single value
    ])
    return model

# Instantiate the actor and critic models
actor = create_actor()
critic = create_critic()

# Section 3: Design Reward Function
# ---------------------------------

# Compute the reward based on the current state, next state, and energy consumption
def compute_reward(state, next_state, energy_consumption, target_temp=22):
    # Comfort penalty: sum of absolute deviations from target temperature
    comfort_penalty = sum(abs(next_state[:9] - target_temp))
    
    # Energy savings: difference in energy consumption
    energy_savings = state[-1] - energy_consumption
    
    # Combine comfort penalty and energy savings to compute the reward
    reward = -comfort_penalty + energy_savings
    return reward

# Section 4: Environment Solution
# -------------------------------

# Define the environment for the building
class BuildingEnv:
    def __init__(self, data, target_temp=22):
        self.data = data
        self.target_temp = target_temp
        self.current_state = None
        self.reset()
    
    # Reset the environment to a random initial state
    def reset(self):
        self.current_state = self.data.sample(1).values[0]
        return self.current_state
    
    # Take a step in the environment based on the action
    def step(self, action):
        next_state = self.current_state.copy()
        if action == 0:
            next_state[:9] -= 1
        elif action == 2:
            next_state[:9] += 1
        
        energy_consumption = next_state[-1]
        reward = compute_reward(self.current_state, next_state, energy_consumption, self.target_temp)
        self.current_state = next_state
        done = False  # Define your own termination condition
        return next_state, reward, done

# Instantiate the environment with the training data
env = BuildingEnv(X_train)

# Section 5: Train the Model over 500 Episodes
# --------------------------------------------

# Define the optimizer and lists to store losses
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
actor_losses = []
critic_losses = []

# Perform a training step
def train_step(state, action, reward, next_state, done):
    with tf.GradientTape() as tape:
        # Predict action probabilities and value
        action_probs = actor(state, training=True)
        value = critic(state, training=True)
        next_value = critic(next_state, training=True)
        
        # Compute advantage
        advantage = reward + (1 - done) * next_value - value
        
        # Compute actor loss
        action_log_probs = tf.math.log(action_probs[0, action])
        actor_loss = -action_log_probs * advantage
        
        # Compute critic loss
        critic_loss = advantage ** 2
        
        # Total loss
        total_loss = actor_loss + critic_loss
    
    # Apply gradients to update the model parameters
    grads = tape.gradient(total_loss, actor.trainable_variables + critic.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables + critic.trainable_variables))
    
    # Record losses for analysis
    actor_losses.append(actor_loss.numpy())
    critic_losses.append(critic_loss.numpy())

# Train the model over a specified number of episodes
num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action_probs = actor(np.expand_dims(state, axis=0))
        action = np.random.choice([0, 1, 2], p=action_probs.numpy()[0])
        next_state, reward, done = env.step(action)
        train_step(np.expand_dims(state, axis=0), action, reward, np.expand_dims(next_state, axis=0), done)
        state = next_state

# Section 6: Evaluate the Performance of the Model
# ------------------------------------------------

# Evaluate the model's performance over a number of episodes
def evaluate_model(env, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action_probs = actor(np.expand_dims(state, axis=0))
            action = np.argmax(action_probs.numpy()[0])
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward
    return total_rewards / num_episodes

# Instantiate the environment with the testing data and evaluate the model
test_env = BuildingEnv(X_test)
average_reward = evaluate_model(test_env)
print(f'Average Reward: {average_reward}')

# Section 7: Provide Graphs Showing the Convergence of the Actor and Critic Losses
# -------------------------------------------------------------------------------

# Plot the actor and critic losses over the training episodes
plt.plot(actor_losses, label='Actor Loss')
plt.plot(critic_losses, label='Critic Loss')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Section 8: Plot the Learned Policy
# ----------------------------------

# Plot the learned policy by visualizing action probabilities for different states
def plot_policy(env, num_samples=100):
    states = []
    actions = []
    for _ in range(num_samples):
        state = env.reset()
        action_probs = actor(np.expand_dims(state, axis=0)).numpy()[0]
        states.append(state)
        actions.append(action_probs)
    
    states = np.array(states)
    actions = np.array(actions)
    
    for i in range(3):
        sns.scatterplot(x=states[:, 0], y=actions[:, i], label=f'Action {i}')
    plt.xlabel('Temperature (T1)')
    plt.ylabel('Action Probability')
    plt.legend()
    plt.show()

# Plot the policy using the testing environment
plot_policy(test_env)

# Section 9: Provide an Analysis on Energy Consumption
# ----------------------------------------------------

# Compare energy consumption before and after training the model
def compare_energy_consumption(env, num_episodes=100):
    initial_energy = 0
    final_energy = 0
    for episode in range(num_episodes):
        state = env.reset()
        initial_energy += state[-1]
        done = False
        while not done:
            action_probs = actor(np.expand_dims(state, axis=0))
            action = np.argmax(action_probs.numpy()[0])
            next_state, reward, done = env.step(action)
            state = next_state
        final_energy += state[-1]
    
    initial_energy /= num_episodes
    final_energy /= num_episodes
    return initial_energy, final_energy

# Calculate and print the initial and final energy consumption
initial_energy, final_energy = compare_energy_consumption(test_env)
print(f'Initial Energy Consumption: {initial_energy}')
print(f'Final Energy Consumption: {final_energy}')
