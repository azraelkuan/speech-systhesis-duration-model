import torch
import os
from torch.autograd import Variable
import argparse
from train import RnnModel
from load_data import load_data
from torch.utils.data import DataLoader, Dataset


# parameters
batch_size = 4
ssp = 'VCC2SM1'
tsp = 'VCC2TM1'
scp_dir = './vcc_scp'
dur_dir = '../dur'
mono_list = 'mono.all.list'


class VCDataSet(Dataset):

    def __init__(self, vcc_data, vcc_length, vcc_wav_ids, transform=False):

        self.vcc_data = vcc_data
        self.vcc_length = vcc_length
        self.vcc_wav_ids = vcc_wav_ids
        self.transform = transform

    def __len__(self):
        return len(self.vcc_data)

    def __getitem__(self, idx):
        sample = {"data": self.vcc_data[idx], "length": self.vcc_length[idx], 'wav_id': self.vcc_wav_ids[idx]}
        if self.transform:
            sample = self.to_tensor(sample)
        return sample

    def to_tensor(self, sample):
        data, length, wav_id = sample['data'], sample['length'], sample['wav_id']

        return {'data': torch.from_numpy(data).float(),
                'length': length,
                'wav_id': wav_id}


def get_args():
    parser = argparse.ArgumentParser(description='duration rnn model')
    parser.add_argument('--model', type=str, default=None, help="the generate model by training")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="the batch size of training")
    parser.add_argument('--ssp', type=str, default=ssp, help="source speaker")
    parser.add_argument('-tsp', type=str, default=tsp, help='target speaker')
    parser.add_argument('--scp_dir', type=str, default=scp_dir, help='the scp dir, include train test dev')
    parser.add_argument('--dur_dir', type=str, default=dur_dir, help='dur dir for training')
    parser.add_argument('--mono_list', type=str, default=mono_list, help='all the mono list')

    return parser.parse_args()


def main():
    phone_level_dur = "../baseline/phone_level/dur/{}.txt"
    args = get_args()
    test_scp = args.scp_dir + "/test.scp"

    os.system("mkdir -p result/dur")

    test_data_x, test_data_y, test_length, test_wav_ids = \
        load_data("vcc", test_scp, args.dur_dir, args.mono_list, 'test')

    vcc_test_data_set = VCDataSet(test_data_x, test_length, test_wav_ids, transform=True)
    vcc_test_loader = DataLoader(dataset=vcc_test_data_set, batch_size=args.batch_size, shuffle=False)

    if args.model is None:
        raise ValueError("the model muse not be None")

    rnn_model = torch.load(args.model)

    for _, tmp in enumerate(vcc_test_loader):
        data = Variable(tmp['data']).cuda()

        length = list(tmp['length'])
        wav_id = tmp['wav_id']
        outputs, length1 = rnn_model(data, length)
        new_wav_id = []
        max_length = max(length1)

        for i in range(len(length1)):
            j = length.index(length1[i])
            new_wav_id.append(wav_id[j])

        for i, each_length in enumerate(length1):
            tmp_outputs = outputs[i*max_length:(i+1)*max_length]
            cur_wav_id = new_wav_id[i]
            cur_pl_dur = phone_level_dur.format(cur_wav_id)
            frames_list = []
            with open(cur_pl_dur, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    frames_list.append(int(line.strip()))

            with open("result/dur/{}.txt".format(cur_wav_id), 'w', encoding='utf-8') as f:
                f.write(str(frames_list[0]) + "\n")
                for j, x in enumerate(tmp_outputs[:each_length].cpu().data.numpy()):
                    x = int(x)
                    # print(x, frames_list[j+1])
                    # if x/frames_list[j+1] < 0.6 or frames_list[j+1]/x < 0.6:
                    #     f.write(str(frames_list[j+1]) + "\n")
                    #     print("phone level")
                    # else:
                    f.write(str(x) + "\n")
                        # print("origin")
                f.write(str(frames_list[-1]) + "\n")


if __name__ == '__main__':
    main()
    
    


