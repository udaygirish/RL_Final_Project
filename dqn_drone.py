import setup_path
import gym
import airgym
import time
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement , BaseCallback
import imageio
import numpy as np
import os
from airgym.envs.custom_policy_sb import CustomCombinedExtractor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from torchsummary import summary

CHECKPOINT_PATH = "./models/dqn_airsim_drone_1000_steps.zip"
# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.7,
                image_shape=(84, 84, 1),
            )
        )
    ]
)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)



print("Creating Custom Policy")

# airgym:airsim-drone-sample-v0
# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
#"CnnPolicy"
model = DQN(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0005,  # Increased learning rate
    verbose=1,
    batch_size=32,
    train_freq=2,  # Increased training frequency
    target_update_interval=10000,
    learning_starts=50, # 10000
    buffer_size=500000,  # Increased buffer size
    max_grad_norm=10,
    exploration_fraction=0.1,  # Increased exploration fraction
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
    seed = 42
)

print("====================================="*5)
print("MODEL CREATED")
print("MODEL ARCHITECTURE: ", model.policy)
print("====================================="*5)

print("MODEL TRAINING")

# Load checkpoint
#CHECKPOINT_PATH = None
if CHECKPOINT_PATH is not None:
    print("Loaded Pretrained Checkpoints")
    model = DQN.load(CHECKPOINT_PATH, env=env)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []

# Stop training if there is no improvement after more than 3 evaluations
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
    callback_after_eval=stop_train_callback,
)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


save_on_best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="models/")
# Stops training when the model reaches the maximum number of episodes
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100, verbose=1)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="dqn_airsim_drone",
                                         save_replay_buffer=True, save_vecnormalize=True)


callbacks.append(eval_callback)
callbacks.append(checkpoint_callback)
#callbacks.append(save_on_best_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e5,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    progress_bar=True,
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_drone_policy")


# Save the images as a gif
images = []
obs = model.env.reset()
img = model.env.render(mode="rgb_array")
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render(mode="rgb_array")

imageio.mimsave("drone_dqn.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

