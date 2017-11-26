import struct
import numpy as np
import os
import wave
from scipy.interpolate import interp1d


wav_path_template = "/speechlab/users/jhl00/vc_JDGMM/data/wav/{}/{}.wav"
out_template = "./phone_level/{}/{}.{}"
lf0_dim = 1
mgc_dim = 150
bap_dim = 5

syn = "/speechlab/users/bc299/tools/aitts_syn_tools/synthesis"

os.system("mkdir -p phone_level/lf0")
os.system("mkdir -p phone_level/bap")
os.system("mkdir -p phone_level/mgc")


def get_phone_scale(source_speaker, target_speaker, dur_dir, scp):

    with open(scp, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        source_phone_dict = {}
        target_phone_dict = {}
        phone_set = set()

        for line in lines:
            wav_id = line.strip()

            source_dur_file = dur_dir + "/{}_{}.txt".format(source_speaker, wav_id)
            target_dur_file = dur_dir + "/{}_{}.txt".format(target_speaker, wav_id)

            with open(source_dur_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            with open(target_dur_file, 'r', encoding='utf-8') as f:
                target_lines = f.readlines()

            for each_line in source_lines:
                each_line = each_line.strip().split()
                phone = each_line[0]
                frames = int(each_line[1])
                if phone in source_phone_dict:
                    source_phone_dict[phone] += frames
                else:
                    source_phone_dict[phone] = frames
                phone_set.add(phone)

            for each_line in target_lines:
                each_line = each_line.strip().split()
                phone = each_line[0]
                frames = int(each_line[1])
                if phone in target_phone_dict:
                    target_phone_dict[phone] += frames
                else:
                    target_phone_dict[phone] = frames
                phone_set.add(phone)
    phone_scale = {}
    for each_phone in list(phone_set):
        phone_scale[each_phone] = round(target_phone_dict[each_phone] / source_phone_dict[each_phone], 2)

    return phone_scale


def convert(source_speaker, target_speaker, dur_dir, test_scp, all_scp, feature_dir_template):
    phone_scale = get_phone_scale(source_speaker, target_speaker, dur_dir, all_scp)
    with open(test_scp, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            wav_id = line.strip()
            print("wav id :{}".format(wav_id))
            source_dur_file = dur_dir + "/{}_{}.txt".format(source_speaker, wav_id)
            with open(source_dur_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            target_dur_file = dur_dir + "/{}_{}.txt".format(target_speaker, wav_id)
            with open(target_dur_file, 'r', encoding='utf-8') as f:
                target_lines = f.readlines()

            # convert feature
            lf0_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dlf0/{}.lf0".format(wav_id)
            out_lf0_file = out_template.format("lf0", wav_id, "lf0")
            convert_feature(lf0_dim, lf0_file_path, out_lf0_file, source_lines, target_lines, phone_scale)

            mgc_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dmgc/{}.mgc".format(wav_id)
            out_mgc_file = out_template.format("mgc", wav_id, "mgc")
            convert_feature(mgc_dim, mgc_file_path, out_mgc_file, source_lines, target_lines, phone_scale)

            bap_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dbap/{}.bap".format(wav_id)
            out_bap_file = out_template.format("bap", wav_id, "bap")
            convert_feature(bap_dim, bap_file_path, out_bap_file, source_lines, target_lines, phone_scale)

            # syn wav
            out_wav_file = out_template.format("wav", wav_id, "wav")
            os.system("mkdir -p overall_feature/wav/")
            os.system("%s -mdim 1 -ford 1 -mord 24 -aord 5 -r 16000 -T 1 %s %s -apf %s %s"
                      % (syn, out_lf0_file, out_mgc_file, out_bap_file, out_wav_file))


def convert_feature(dim, input_file, output_file, source_lines, target_lines, phone_scale):
    with open(input_file, 'rb') as f:
        data = np.fromfile(f, dtype='<f', count=-1, sep='')
    frames = len(data) // dim
    # need to align the frames
    source_frames = 0
    frames_list = []
    origin_frames_list = []
    for each_line in source_lines:
        each_line = each_line.strip().split()
        source_frames += int(each_line[1])

    extra_frames = frames - source_frames
    assert extra_frames >= 0
    begin_add_frames = extra_frames // 2
    end_add_frames = extra_frames - begin_add_frames

    for i, each_line in enumerate(source_lines):
        each_line = each_line.strip().split()
        phone = each_line[0]
        each_frames = int(each_line[1])
        if i == 0:
            each_frames += begin_add_frames
        if i == (len(source_lines) - 1):
            each_frames += end_add_frames
        origin_frames_list.append(each_frames)
        tmp_frames = int(phone_scale[phone] * each_frames)
        each_frames = tmp_frames
        if each_frames == 0:
            each_frames += 1
        frames_list.append(each_frames)

    mse = 0

    if len(target_lines) == len(frames_list):
        for real_line, predict in zip(target_lines, frames_list):
            real_line = real_line.strip().split()
            real = int(real_line[1])
            mse += (real - predict) ** 2
        print(mse / len(target_lines))
    if "lf0" in input_file:
        wav_id = output_file.split('/')[-1].split('.')[0]
        os.system('mkdir -p phone_level/dur')
        out_dur_file = "phone_level/dur/{}.txt".format(wav_id)
        out_f = open(out_dur_file, 'w', encoding="utf-8")
        for each in frames_list:
            out_f.write(str(each) + "\n")
        out_f.close()

    new_frames = sum(frames_list)
    assert sum(origin_frames_list) == frames

    data = np.array(data).reshape(frames, dim).T
    new_data = np.zeros(dim * new_frames).reshape(dim, new_frames)

    for i in range(0, dim):
        start_frames = 0
        origin_start_frames = 0
        for j in range(len(frames_list)):
            end_frames = start_frames + frames_list[j]
            origin_end_frames = origin_start_frames + origin_frames_list[j]

            time_axis = np.arange(origin_end_frames - origin_start_frames)
            interp = interp1d(time_axis, data[i][origin_start_frames:origin_end_frames],
                              kind='linear', bounds_error=False, fill_value=0)

            time_axis = np.arange(frames_list[j]) * origin_frames_list[j] / frames_list[j]

            new_data[i][start_frames:end_frames] = new_data[i][start_frames:end_frames] + interp(time_axis)
            start_frames = end_frames
            origin_start_frames = origin_end_frames

    new_data = new_data.T.reshape(new_frames * dim, 1)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('<%df' % len(new_data), *new_data))


if __name__ == '__main__':
    source_speaker = "vcc2sm1".upper()
    target_speaker = "vcc2tm1".upper()
    scp_dir = "../train_pytorch/vcc_scp/"
    feature_dir_template = "/speechlab/users/jhl00/syn24/mlpg_wav/{}_{}_nop/"
    convert(source_speaker, target_speaker, "../dur/", scp_dir+'test.scp', scp_dir+"all.scp", feature_dir_template)


