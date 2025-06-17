from pathlib import Path
from typing import Optional, Tuple
import dm_env.specs
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm

import sac
import specs
import replay
import dm_env

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    # environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    environment_name: str = "RoboPianist-repertoire-150-PWaltzOp69No2-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = "/home/zhou/videos"
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: sac.SACConfig = sac.SACConfig()


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}

def compute_total_dim(spec):
    """递归计算嵌套观测空间的总维度"""
    if isinstance(spec, dict):
        # 处理字典类型的观测空间
        return sum(compute_total_dim(v) for v in spec.values())
    elif isinstance(spec, (list, tuple)):
        # 处理列表或元组类型的观测空间
        return sum(compute_total_dim(v) for v in spec)
    else:
        # 处理单个观测规格（如BoundedArray）
        return np.prod(spec.shape)  # 计算多维数组的总元素数



def get_env(args: Args, record_dir: Optional[Path] = None):
    env = suite.load(
        environment_name=args.environment_name,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    
    # timestep = env.reset()
    # print(f"Initial Observation (before wrapping): {timestep.observation}")
    # if 'goal' in timestep.observation:
    #     print(f"Goal value: {timestep.observation['goal']}")
    #     print(f"Goal value: {timestep.observation['goal'].shape}")

    # print("\n===== 原始环境规格 =====")
    # original_observation_spec = env.observation_spec()
    # if "goal" in original_observation_spec:
    #     goal_dim = original_observation_spec["goal"].shape[-1]
    #     # print(goal_dim)
    # else:
    #     raise ValueError("原始观测规范中不包含 'goal' 字段。")
    # print("观测空间:", compute_total_dim(env.observation_spec()))
    # print("观测空间:", env.observation_spec())
    # print("动作空间:", env.action_spec().shape)
    # print("动作空间:", env.action_spec())

    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        # print("\n===== PianoSoundVideoWrapper后环境规格 =====")
        # print("观测空间:", compute_total_dim(env.observation_spec()))
        # print("观测空间:", env.observation_spec())
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        # print("\n===== EpisodeStatisticsWrapper后环境规格 =====")
        # print("观测空间:", compute_total_dim(env.observation_spec()))
        # print("观测空间:", env.observation_spec())
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
        # print("\n===== MidiEvaluationWrapper后环境规格 =====")
        # print("观测空间:", compute_total_dim(env.observation_spec()))
        # print("观测空间:", env.observation_spec())
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
        # print("\n===== EpisodeStatisticsWrapper后环境规格 =====")
        # print("观测空间:", compute_total_dim(env.observation_spec()))
        # print("观测空间:", env.observation_spec())
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
        # print("\n===== ObservationActionRewardWrapper后环境规格 =====")
        # print("观测空间:", compute_total_dim(env.observation_spec()))
        # print("观测空间:", env.observation_spec())
    env = wrappers.ConcatObservationWrapper(env)
    # concat_observation_spec = env.observation_spec()
    # if isinstance(concat_observation_spec, dm_env.specs.BoundedArray):
    #     sorted_fields = sorted(original_observation_spec.keys())
    #     # 计算每个字段的起始位置
    #     start_positions = {}
    #     current_position = 0
    #     for field in sorted_fields:
    #         start_positions[field] = current_position
    #         current_position += original_observation_spec[field].shape[-1]
    #     # 确定 goal 在连接后的张量中的起始位置
    #     concat_goal_start_position = start_positions["goal"]
    #     concat_goal_end_position = concat_goal_start_position + goal_dim
    # else:
    #     raise ValueError("ConcatObservationWrapper 的观测规范不是 BoundedArray 类型。")
    # print(f"Goal 位于最终观测空间的 {concat_goal_start_position} 到 {concat_goal_end_position} 维。")
    # print("\n===== ConcatObservationWrapper后环境规格 =====")
    # print("观测空间:", compute_total_dim(env.observation_spec()))
    # print("观测空间:", env.observation_spec())
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
        # print("\n===== FrameStackingWrapper后环境规格 =====")
        # print("观测空间:", compute_total_dim(env.observation_spec()))
        # print("观测空间:", env.observation_spec())
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    # print("\n===== CanonicalSpecWrapper后环境规格 =====")
    # print("观测空间:", compute_total_dim(env.observation_spec()))
    # print("观测空间:", env.observation_spec())
    env = wrappers.SinglePrecisionWrapper(env)
    # print("\n===== SinglePrecisionWrapper后环境规格 =====")
    # print("观测空间:", compute_total_dim(env.observation_spec()))
    # print("观测空间:", env.observation_spec())
    env = wrappers.DmControlWrapper(env)

    # print("\n===== 包装后环境规格 =====")
    # print("观测空间:", compute_total_dim(env.observation_spec()))
    # print("观测空间:", env.observation_spec())
    # print("动作空间:", env.action_spec().shape)

    return env


def main(args: Args) -> None:
    # print(f"reduced_action_space: {args.reduced_action_space}")
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.environment_name}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )

    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval")

    spec = specs.EnvironmentSpec.make(env)
    # print(spec.observation)

    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )

    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()
    for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        # Act.
        if i < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)
        
        # print(f"Step {i}: Action = {action}")
        # print(f"Step {i}: Action dimension = {action.shape}")

        # Observe.
        timestep = env.step(action)

        print(f"Step {i}: Observation = {timestep.observation}")
        # print(f"Step {i}: Goal = {timestep.observation[:88]}")
        print(f"Step {i}: Observation dimension = {timestep.observation.shape}")

        replay_buffer.insert(timestep, action)

        # Reset episode.
        if timestep.last():
            wandb.log(prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # Train.
        if i >= args.warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    wandb.log(prefix_dict("train", metrics), step=i)

        # Eval.
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                timestep = eval_env.reset()
                while not timestep.last():
                    timestep = eval_env.step(agent.eval_actions(timestep.observation))
            log_dict = prefix_dict("eval", eval_env.get_statistics())
            music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
            wandb.log(log_dict | music_dict, step=i)
            video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
            wandb.log({"video": video, "global_step": i})
            eval_env.latest_filename.unlink()

        if i % args.log_interval == 0:
            wandb.log({"train/fps": int(i / (time.time() - start_time))}, step=i)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
