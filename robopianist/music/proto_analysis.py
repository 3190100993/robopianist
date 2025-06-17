import midi_file  # 假设 midi_file 模块存在
from robopianist import suite
from pathlib import Path
import numpy as np

# proto_file_path = Path("/home/zhou/robopianist/robopianist/music/data/pig_single_finger/waltz_op_69_no_2-1.proto")
# try:
#     # 调用 from_file 方法加载 .proto 文件
#     midi = midi_file.MidiFile.from_file(proto_file_path)
#     print("成功解析 .proto 文件为 MidiFile 对象。")
#     print(midi)
#     # 这里你可以根据需要对 midi 对象进行进一步操作
#     # 例如，如果需要获取 NoteSequence 对象
#     seq = midi.seq  # 假设 MidiFile 类有 seq 属性来存储 NoteSequence
#     print("成功获取 NoteSequence 对象。")
#     print(seq)
# except ValueError as e:
#     print(f"不支持的文件扩展名: {e}")
# except RuntimeError as e:
#     print(f"解析 MIDI 文件时出错: {e}")

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

# env = suite.load('RoboPianist-repertoire-150-PolonaiseFantasieOp61-v0')
env = suite.load(
        environment_name="RoboPianist-repertoire-150-WaltzOp69No2-v0",
        seed=42,
        stretch=1.0,
        shift=0,
        task_kwargs=dict(
            n_steps_lookahead=0,
            trim_silence=False,
            gravity_compensation=False,
            reduced_action_space=False,
            control_timestep=0.05,
            wrong_press_termination=False,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            primitive_fingertip_collisions=False,
            change_color_on_activation=True,
        ),
    )
# print(env.observation_spec())
# print(env.action_spec())
print("观测空间:", compute_total_dim(env.observation_spec()))
print("观测空间:", env.observation_spec())
print("动作空间:", env.action_spec().shape)
print("动作空间:", env.action_spec())