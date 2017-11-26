import numpy as np


def compute_loss(source_speaker, target_speaker, dur_dir, wav_id):

        input_dur_file = dur_dir + "/cmu_us_arctic_{}_{}.txt".format(source_speaker, wav_id)
        output_dur_file = dur_dir + "/cmu_us_arctic_{}_{}.txt".format(target_speaker, wav_id)

        with open(input_dur_file, 'r', encoding='utf-8') as f:
            input_lines = f.readlines()
        with open(output_dur_file, 'r', encoding='utf-8') as f:
            output_lines = f.readlines()

        # remove sil at the begin and the end
        input_lines = input_lines[1:-2]
        output_lines = output_lines[1:-2]

        if len(input_lines) != len(output_lines):
            # print(len(input_lines), len(output_lines), wav_id)
            pass
        else:
            tmp_x = []
            tmp_y = []
            for each_line in input_lines:
                frames = int(each_line.strip().split()[1])
                if frames >=200:
                    frames = 199
                tmp_x.append(frames)

            for each_line in output_lines:
                frames = int(each_line.strip().split()[1])
                if frames >= 200:
                    frames = 199
                tmp_y.append(frames)

            tmp_x = np.array(tmp_x).T
            tmp_y = np.array(tmp_y).T
            loss = np.sum(np.power(tmp_x - tmp_y, 2))
            return loss