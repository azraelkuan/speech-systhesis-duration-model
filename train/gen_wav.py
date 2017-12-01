import struct
import numpy as np
import os
import wave
from scipy.interpolate import interp1d
import random
# from matplotlib import pyplot as plt


wav_path_template = "/speechlab/users/jhl00/vc_JDGMM/data/wav/{}/{}.wav"
out_template = "./result/{}/{}.{}"
lf0_dim = 1
mgc_dim = 25
bap_dim = 5

syn = "/speechlab/users/bc299/tools/aitts_syn_tools/synthesis"

os.system("mkdir -p result/lf0")
os.system("mkdir -p result/bap")
os.system("mkdir -p result/mgc")


def convert(source_speaker, target_speaker, dur_dir, test_scp, feature_dir_template):

    with open(test_scp, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            wav_id = line.strip()
            print("wav id :{}".format(wav_id))
            source_dur_file = dur_dir + "/{}_{}.txt".format(source_speaker, wav_id)
            predict_dur_file = "result/dur/{}.txt".format(wav_id)
            if os.path.isfile(predict_dur_file):
                predict_frames = []
                with open(source_dur_file, 'r', encoding='utf-8') as f:
                    source_lines = f.readlines()
                with open(predict_dur_file, 'r', encoding='utf-8') as f:
                    predict_lines = f.readlines()
                    for predict_line in predict_lines:
                        predict_line = predict_line.strip()
                        predict_frames.append(int(predict_line))
                    # convert feature
                lf0_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dlf0/{}.lf0".format(wav_id)
                out_lf0_file = out_template.format("lf0", wav_id, "lf0")
                convert_feature(lf0_dim, lf0_file_path, out_lf0_file, source_lines, predict_frames)

                mgc_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/VCmgc/{}.mgc".format(wav_id)
                out_mgc_file = out_template.format("mgc", wav_id, "mgc")
                convert_feature(mgc_dim, mgc_file_path, out_mgc_file, source_lines, predict_frames)

                bap_file_path = feature_dir_template.format(source_speaker, target_speaker) + "/dbap/{}.bap".format(wav_id)
                out_bap_file = out_template.format("bap", wav_id, "bap")
                convert_feature(bap_dim, bap_file_path, out_bap_file, source_lines, predict_frames)

                # syn wav
                out_wav_file = out_template.format("wav", wav_id, "wav")
                os.system("mkdir -p result/wav/")
                cmd = "%s -b 0.4 -ford 1 -mord 24 -aord 5 -r 16000 -T 1 %s %s -apf %s %s" % \
                      (syn, out_lf0_file, out_mgc_file, out_bap_file, out_wav_file)
                print(cmd)
                os.system(cmd)


def convert_feature(dim, input_file, output_file, source_lines, predict_frames):
    with open(input_file, 'rb') as f:
        data = np.fromfile(f, dtype='<f', count=-1, sep='')
    frames = len(data) // dim
    # need to align the frames
    source_frames = 0
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

    # add the sil of head and tail
    # if len(predict_frames) == len(origin_frames_list) - 2:
    #     predict_frames.insert(0, origin_frames_list[0])
    #     predict_frames.append(origin_frames_list[-1])
    
    # limit the predict frames
    for i in range(len(predict_frames)):
        if predict_frames[i] < origin_frames_list[i] * 1 / 3:
            predict_frames[i] = int(origin_frames_list[i] * 2 / 3)
        if predict_frames[i] > origin_frames_list[i] * 3:
            predict_frames[i] = int(origin_frames_list[i] * 1.5)

    assert len(predict_frames) == len(origin_frames_list)

    new_frames = sum(predict_frames)
    assert sum(origin_frames_list) == frames

    data = data.reshape(frames, dim).T
    new_data = np.zeros(dim * new_frames).reshape(dim, new_frames)

    for i in range(0, dim):
        if "lf0" in output_file:
            data[i] = np.exp(data[i])

        start_frames = 0
        origin_start_frames = 0
        for j in range(0, len(predict_frames), 3):
            if j+3 <= len(predict_frames):
                total_predict_frame = sum(predict_frames[j:j+3])
                total_origin_frame = sum(origin_frames_list[j:j+3])
            else:
                total_predict_frame = sum(predict_frames[j:])
                total_origin_frame = sum(origin_frames_list[j:])

            end_frames = start_frames + total_predict_frame
            origin_end_frames = origin_start_frames + total_origin_frame
            time_axis = np.arange(total_origin_frame)
            interp = interp1d(time_axis, data[i][origin_start_frames:origin_end_frames], kind="slinear")

            if total_predict_frame > total_origin_frame:
                step = round((total_origin_frame -1) / (total_predict_frame-1), 2)
                time_axis = np.arange(0, total_origin_frame, step)
                if time_axis.shape[0] > total_predict_frame:
                    time_axis = time_axis[0:total_predict_frame]
                    time_axis[-1] = total_origin_frame - 1
                if time_axis[-1] > total_origin_frame - 1:
                    time_axis[-1] = total_origin_frame - 1
            else:
                time_axis = np.arange(0, total_predict_frame)
            try:
                new_data[i][start_frames:end_frames] = interp(time_axis)
            except:
                print(total_origin_frame, total_predict_frame)
                print(time_axis)
            # print(end_frames - start_frames, origin_end_frames - origin_start_frames)
            start_frames = end_frames
            origin_start_frames = origin_end_frames

        if "lf0" in output_file:
            for ii in range(new_frames):
                if int(new_data[i][ii]) == 0:
                    new_data[i][ii] = -1e10
                else:
                    new_data[i][ii] = np.log(new_data[i][ii])

    new_data = new_data.T.reshape(new_frames * dim, 1)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('<%df' % len(new_data), *new_data))


if __name__ == '__main__':
    feature_dir_template = "../data/{}_{}_all/"
    convert("VCC2SM1", 'VCC2TM1', "../dur/", './vcc_scp/test.scp', feature_dir_template)
