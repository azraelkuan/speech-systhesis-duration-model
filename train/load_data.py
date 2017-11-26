import difflib

import numpy as np
import math


def load_mono_list(mono_list):
    mono = []
    with open(mono_list, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            mono.append(line)
    return mono


def load_data(data_name, scp, dur_dir, mono_list, mode='train'):
    mono = load_mono_list(mono_list)
    source_speakers = []
    if data_name == "arctic":
        source_speakers = ['clb', 'slt']
    if data_name == "vcc":
        source_speakers = ['VCC2SM1', 'VCC2TM1']

    assert len(source_speakers) >= 2
    with open(scp, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        X = []
        Y = []
        length = []
        wav_ids = []
        max_phone_size = 128

        for source_speaker in source_speakers:
            index = source_speakers.index(source_speaker)
            for target_speaker in source_speakers[index+1:]:
                for line in lines:
                    wav_id = line.strip()
                    input_dur_file = dur_dir + "/{}_{}.txt".format(source_speaker, wav_id)
                    output_dur_file = dur_dir + "/{}_{}.txt".format(target_speaker, wav_id)

                    with open(input_dur_file, 'r', encoding='utf-8') as f:
                        input_lines = f.readlines()
                    with open(output_dur_file, 'r', encoding='utf-8') as f:
                        output_lines = f.readlines()

                    if mode == 'train':
                        if len(input_lines) != len(output_lines):
                            source_phones_list = []
                            target_phones_list = []
                            for line in input_lines:
                                source_phones_list.append(line.strip().split()[0])
                            for line in output_lines:
                                target_phones_list.append(line.strip().split()[0])
                            source_string = " ".join(source_phones_list)
                            target_string = " ".join(target_phones_list)

                            s = difflib.SequenceMatcher(None, source_string, target_string)
                            all_ops = []
                            for opcode in s.get_opcodes():
                                all_ops.append(opcode)

                            for opcode in reversed(all_ops):
                                if opcode[0] == 'insert':
                                    start = opcode[3]
                                    index = 0
                                    for i, _ in enumerate(target_string):
                                        if _ == " ":
                                            index += 1
                                        if i == start:
                                            output_lines.pop(index)
                                if opcode[0] == "delete":
                                    start = opcode[1]
                                    index = 0
                                    for i, _ in enumerate(source_string):
                                        if _ == " ":
                                            index += 1
                                        if i == start:
                                            input_lines.pop(index)

                    # remove sil at the begin and the end
                    input_lines = input_lines[1:-1]
                    output_lines = output_lines[1:-1]

                    tmp_x = []
                    tmp_y = []
                    for each_line in input_lines:
                        each_phone = each_line.strip().split()[0]
                        frames = int(each_line.strip().split()[1])

                        tmp = [0 for _ in range(len(mono))]
                        tmp[mono.index(each_phone)] = 1

                        tmp_id = [0 for _ in range(len(source_speakers))]
                        tmp_id[source_speakers.index(source_speaker)] = 1
                        tmp_id[source_speakers.index(target_speaker)] = 1
                        tmp.extend(tmp_id)
                        tmp.append(frames)
                        tmp_x.append(tmp)

                    for each_line in output_lines:
                        frames = int(each_line.strip().split()[1])
                        tmp_y.append(frames)

                    tmp_x = np.array(tmp_x)
                    length.append(len(tmp_x))

                    # pad tmp_x to fixed length
                    tmp_x = np.pad(tmp_x, [[0, max_phone_size-tmp_x.shape[0]], [0, 0]], mode='constant')
                    tmp_y = np.array(tmp_y)
                    tmp_y = np.pad(tmp_y, [[0, max_phone_size-tmp_y.shape[0]]], mode='constant')
                    X.append(tmp_x)
                    Y.append(tmp_y)
                    wav_ids.append(wav_id)
                    # print(wav_id, len(input_lines), len(output_lines))
    X = np.array(X)
    Y = np.array(Y)
    length = np.array(length)
    return X, Y, length, wav_ids


# if __name__ == '__main__':
#     load_data('vcc', 'vcc_scp/train.scp', '../dur', 'mono.all.list')

