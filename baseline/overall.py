import wave
import numpy as np
from scipy.interpolate import interp1d
import struct
import os

"""
整体时长变换
"""

wav_path_template = "/speechlab/users/jhl00/vc_JDGMM/data/wav/{}/{}.wav"
out_template = "./overall/{}/{}.{}"

lf0_dim = 1
mgc_dim = 150
bap_dim = 5

syn = "/speechlab/users/bc299/tools/aitts_syn_tools/synthesis"

os.system("mkdir -p overall/lf0")
os.system("mkdir -p overall/bap")
os.system("mkdir -p overall/mgc")


def get_wav_ids(scp):
    wav_ids = set()
    with open(scp, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            wav_id = line.strip()
            wav_ids.add(wav_id)
    return list(wav_ids)


def compute_duration_scale(source_speaker, target_speaker, scp):
    source_time = 0
    target_time = 0
    wav_ids = get_wav_ids(scp)
    for wav_id in wav_ids:
        source_wav = wav_path_template.format(source_speaker, wav_id)
        target_wav = wav_path_template.format(target_speaker, wav_id)
        source_frames = wave.open(source_wav, 'r').getnframes()
        target_frames = wave.open(target_wav, 'r').getnframes()
        source_time += source_frames
        target_time += target_frames
    scale = round(target_time / source_time, 2)
    for wav_id in wav_ids:
        target_wav = wav_path_template.format(target_speaker, wav_id)
        target_frames = wave.open(target_wav, 'r').getnframes()
        tmp_targets = scale * target_frames
        mse = ((target_frames - tmp_targets) / 80)**2
        print(wav_id, mse, target_frames, tmp_targets)
    return scale


def convert(source_speaker, target_speaker, scale, test_scp, feature_dir_template):
    wav_ids = get_wav_ids(test_scp)
    for wav_id in wav_ids:
        # convert feature
        lf0_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dlf0/{}.lf0".format(wav_id)
        out_lf0_file = out_template.format("lf0", wav_id, "lf0")
        convert_feature(lf0_dim, lf0_file_path, out_lf0_file)

        mgc_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dmgc/{}.mgc".format(wav_id)
        out_mgc_file = out_template.format("mgc", wav_id, "mgc")
        convert_feature(mgc_dim, mgc_file_path, out_mgc_file)

        bap_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dbap/{}.bap".format(wav_id)
        out_bap_file = out_template.format("bap", wav_id, "bap")
        convert_feature(bap_dim, bap_file_path, out_bap_file)

        # syn wav
        out_wav_file = out_template.format("wav", wav_id, "wav")
        os.system("mkdir -p overall/wav/")
        os.system("%s -mdim 1 -ford 1 -mord 24 -aord 5 -r 16000 -T 1 %s %s -apf %s %s"
                  % (syn, lf0_file_path, mgc_file_path, bap_file_path, out_wav_file))


def convert_feature(dim, input_file, output_file):
        with open(input_file, 'rb') as f:
            data = np.fromfile(f, dtype='<f', count=-1, sep='')
        frames = len(data) // dim
        new_frames = int(frames * scale)
        data = np.array(data).reshape(frames, dim).T
        new_data = np.zeros(dim * new_frames).reshape(dim, new_frames)

        for i in range(0, dim):
            time_axis = np.arange(frames)
            interp = interp1d(time_axis, data[i], kind='linear', bounds_error=False, fill_value=0)
            time_axis = np.arange(new_frames) * frames / new_frames
            new_data[i] = new_data[i] + interp(time_axis)

        new_data = new_data.T.reshape(new_frames * dim, 1)

        with open(output_file, 'wb') as f:
            f.write(struct.pack('<%df' % len(new_data), *new_data))


if __name__ == '__main__':
    source_speaker = "vcc2sm1".upper()
    target_speaker = "vcc2tm1".upper()
    scp_dir = "../train_pytorch/vcc_scp/"
    scale = compute_duration_scale(source_speaker, target_speaker, scp_dir + "all.scp")
    print(scale)
    feature_dir_template = "/speechlab/users/jhl00/syn24/mlpg_wav/{}_{}_nop/"
    convert(source_speaker, target_speaker, scale, scp_dir + "test.scp", feature_dir_template)

