# @title Run to install robopianist
from IPython.display import clear_output
import subprocess

if subprocess.run("nvidia-smi").returncode:
    raise RuntimeError(
        "Cannot communicate with GPU. "
        "Make sure you are using a GPU Colab runtime. "
        "Go to the Runtime menu and select Choose runtime type."
    )

# Install dependencies.
# %shell bash <(curl -s https://raw.githubusercontent.com/google-research/robopianist/main/scripts/install_deps.sh) --no-soundfonts --no-menagerie
try:
    subprocess.run("bash <(curl -s https://raw.githubusercontent.com/google-research/robopianist/main/scripts/install_deps.sh) --no-soundfonts --no-menagerie", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error installing dependencies: {e}")


print("Installing robopianist...")
# %pip install -q robopianist>=1.0.6
try:
    subprocess.run("pip install -q robopianist>=1.0.6", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error installing robopianist: {e}")

# %env MUJOCO_GL=egl
# 设置环境变量
import os
os.environ["MUJOCO_GL"] = "glfw"

# clear_output()
# %shell echo Installed $(robopianist --version)
# 显示安装的版本
try:
    result = subprocess.run("robopianist --version", shell=True, capture_output=True, text=True)
    print(f"Installed {result.stdout.strip()}")
except subprocess.CalledProcessError as e:
    print(f"Error getting robopianist version: {e}")

# @title All imports required for this tutorial
from IPython.display import HTML
from base64 import b64encode
import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
import subprocess

# @title Helper functions


# Reference: https://stackoverflow.com/a/60986234.
def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return HTML(
        """
  <video controls>
        <source src="%s" type="video/mp4">
  </video>
  """
        % data_url
    )

task = self_actuated_piano.SelfActuatedPiano(
    midi=music.load("TwinkleTwinkleRousseau"),
    change_color_on_activation=True,
    trim_silence=True,
    control_timestep=0.01,
)

env = composer_utils.Environment(
    recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
)

env = PianoSoundVideoWrapper(
    env,
    record_every=1,
    camera_id="piano/back",
    record_dir=".",
)

# # Self-actuated piano task
# action_spec = env.action_spec()
# min_ctrl = action_spec.minimum
# max_ctrl = action_spec.maximum
# print(f"Action dimension: {action_spec.shape}")

# print("Observables:")
# timestep = env.reset()
# dim = 0
# for k, v in timestep.observation.items():
#     print(f"\t{k}: {v.shape} {v.dtype}")
#     dim += np.prod(v.shape)
# print(f"Observation dimension: {dim}")

# class Oracle:
#     def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
#         if timestep.reward is not None:
#             assert timestep.reward == 0
#         # Only grab the next timestep's goal state.
#         goal = timestep.observation["goal"][: task.piano.n_keys]
#         key_idxs = np.flatnonzero(goal)
#         # For goal keys that should be pressed, set the action to the maximum
#         # actuator value. For goal keys that should be released, set the action to
#         # the minimum actuator value.
#         action = min_ctrl.copy()
#         action[key_idxs] = max_ctrl[key_idxs]
#         # Grab the sustain pedal action.
#         action[-1] = timestep.observation["goal"][-1]
#         return action

# policy = Oracle()

# timestep = env.reset()
# while not timestep.last():
#     action = policy(timestep)
#     timestep = env.step(action)

# play_video(env.latest_filename)

# Piano with Shadow Hands
task = piano_with_shadow_hands.PianoWithShadowHands(
    change_color_on_activation=True,
    midi=music.load("TwinkleTwinkleRousseau"),
    trim_silence=True,
    control_timestep=0.05,
    gravity_compensation=True,
    primitive_fingertip_collisions=False,
    reduced_action_space=False,
    n_steps_lookahead=10,
    disable_fingering_reward=False,
    disable_forearm_reward=False,
    disable_colorization=False,
    disable_hand_collisions=False,
    attachment_yaw=0.0,
)

env = composer_utils.Environment(
    task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
)

env = PianoSoundVideoWrapper(
    env,
    record_every=1,
    camera_id="piano/back",
    record_dir=".",
)

env = CanonicalSpecWrapper(env)

action_spec = env.action_spec()
print(f"Action dimension: {action_spec.shape}")

timestep = env.reset()
dim = 0
for k, v in timestep.observation.items():
    print(f"\t{k}: {v.shape} {v.dtype}")
    dim += int(np.prod(v.shape))
print(f"Observation dimension: {dim}")

# Download pretrained policy action sequence.
# %shell wget https://github.com/google-research/robopianist/raw/main/examples/twinkle_twinkle_actions.npy > /dev/null 2>&1
subprocess.run(["wget", "https://github.com/google-research/robopianist/raw/main/examples/twinkle_twinkle_actions.npy","-q"])


class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._idx = 0
        self._actions = np.load("twinkle_twinkle_actions.npy")

    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused.
        actions = self._actions[self._idx]
        self._idx += 1
        return actions
    
policy = Policy()

timestep = env.reset()
while not timestep.last():
    action = policy(timestep)
    timestep = env.step(action)

play_video(env.latest_filename)