import numpy as np
import plotly.graph_objects as go
from robopianist.music import midi_file, library
from note_seq.protobuf import music_pb2


def main():
    # 加载《一闪一闪小星星》的MIDI文件
    # midi = library.twinkle_twinkle_little_star_one_hand()
    # 从 Proto 转 MIDI
    # 创建一个 NoteSequence 对象
    seq = music_pb2.NoteSequence()
    with open("/home/zhou/robopianist/robopianist/music/data/pig_single_finger/arabesque_no_1-1.proto", "rb") as f:
        seq.ParseFromString(f.read())
    
    midi = midi_file.MidiFile(seq=seq)

    # 定义离散时间步长
    dt = 0.1
    # print(midi)

    # 将MIDI文件转换为时间序列表示
    trajectory = midi_file.NoteTrajectory.from_midi(midi, dt)
    print(trajectory)

    # # 打印时间序列
    # print("时间序列表示：")
    # for t, notes in enumerate(trajectory.notes):
    #     active_notes = [note.number for note in notes]
    #     print(f"时间步 {t}: 活动音符 {active_notes}")

    # # 转换为钢琴卷表示
    # piano_roll = trajectory.to_piano_roll()
    # # print(piano_roll.shape)
    # for t, row in enumerate(piano_roll):
    #     active_pitches = np.nonzero(row)[0]
    #     print(f"时间步 {t}: 活跃音符索引 = {active_pitches}")


    # # 绘制交互式钢琴卷图
    # num_timesteps, num_pitches = piano_roll.shape
    # time_axis = np.arange(0, num_timesteps * dt, dt)

    # fig = go.Figure(data=go.Heatmap(
    #     z=piano_roll,
    #     x=time_axis,
    #     y=np.arange(1, num_pitches + 1),
    #     colorscale='gray_r',
    #     hovertemplate='时间: %{x:.2f} 秒<br>音高: %{y}<br>活动: %{z}',
    # ))

    # fig.update_layout(
    #     title='一闪一闪小星星 - 钢琴卷图',
    #     xaxis_title='时间 (秒)',
    #     yaxis_title='音高 (1 - 88)',
    # )

    # fig.show()

if __name__ == "__main__":
    main()
