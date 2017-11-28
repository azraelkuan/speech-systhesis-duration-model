import torch
import argparse
import torch.nn as nn
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from load_data import load_data
from torch.optim.lr_scheduler import ReduceLROnPlateau


# parameters
input_size = 48
hidden_size = 64
num_layers = 1
num_classes = 1
batch_size = 4
num_epochs = 200
learning_rate = 0.01
dropout = 0.5
data_name = 'vcc'
scp_dir = './vcc_scp/'
dur_dir = '../dur'
mono_list = 'mono.all.list'
weight_decay = 0.


def get_args():
    parser = argparse.ArgumentParser(description='duration rnn model')

    parser.add_argument('--hidden_size', type=int, default=hidden_size, help="rnn model hidden size")
    parser.add_argument('--num_layers', type=int, default=num_layers, help="rnn model num layers")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="the batch size of training")
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="the training max epoch")
    parser.add_argument('--lr', type=float, default=learning_rate, help='the training learning rate')
    parser.add_argument('--drop_out', type=float, default=dropout, help='the rnn model drop out')
    parser.add_argument('--data_name', type=str, default=data_name, help="data name")
    parser.add_argument('--scp_dir', type=str, default=scp_dir, help='the scp dir, include train dev')
    parser.add_argument('--dur_dir', type=str, default=dur_dir, help='dur dir for training')
    parser.add_argument('--mono_list', type=str, default=mono_list, help='all the mono list')
    parser.add_argument('--weight_decay', type=float, default=weight_decay, help="weight decay")
    return parser.parse_args()


class VCDataSet(Dataset):

    def __init__(self, vcc_data, vcc_label, vcc_length, vcc_wav_ids, transform=False):

        self.vcc_data = vcc_data
        self.vcc_label = vcc_label
        self.vcc_length = vcc_length
        self.vcc_wav_ids = vcc_wav_ids
        self.transform = transform

    def __len__(self):
        return len(self.vcc_data)

    def __getitem__(self, idx):
        sample = {"data": self.vcc_data[idx], "label": self.vcc_label[idx],
                  "length": self.vcc_length[idx], 'wav_id': self.vcc_wav_ids[idx]}
        if self.transform:
            sample = self.to_tensor(sample)
        return sample

    def to_tensor(self, sample):
        data, label, length, wav_id = sample['data'], sample['label'], sample['length'], sample['wav_id']

        return {'data': torch.from_numpy(data).float(),
                'label': torch.from_numpy(label).float(),
                'length': length,
                'wav_id': wav_id}


class RnnModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RnnModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x, length):
        # init h0, c0
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda()

        x = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)

        out, _ = self.lstm(x, (h0, c0))

        out, new_length = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        
        out = self.fc(out)

        return out, new_length


def test_loss(model, data_loader, criterion):
    loss = 0.
    for i, tmp in enumerate(data_loader):
        data = tmp['data']
        label = tmp['label']
        length = tmp['length']

        max_len = int(torch.max(length))
        data = data[:, :max_len, :]
        label = label[:, :max_len]

        sorted_length, indices = torch.sort(
            length.view(-1), dim=0, descending=True)
        sorted_length = sorted_length.long().numpy()

        data, label = data[indices], label[indices]

        data = Variable(data).cuda()

        outputs, out_length = model(data, sorted_length)

        outputs = outputs.view(label.size(0), -1)

        tmp_loss = criterion(outputs, Variable(label).cuda())

        loss += tmp_loss.data[0]

    return loss / len(data_loader)


def main():
    args = get_args()
    train_scp = args.scp_dir + "/all.scp"
    dev_scp = args.scp_dir + "/dev.scp"
    test_scp = args.scp_dir + "/test.scp"
    training_losses = []
    dev_losses = []

    train_data_x, train_data_y, train_length, train_wav_ids = \
        load_data(args.data_name, train_scp, args.dur_dir, args.mono_list, 'train')
    dev_data_x, dev_data_y, dev_length, dev_wav_ids = \
        load_data(args.data_name, dev_scp, args.dur_dir, args.mono_list, 'train')
    test_data_x, test_data_y, test_length, test_wav_ids = \
        load_data(args.data_name, test_scp, args.dur_dir, args.mono_list, 'train')

    vcc_train_data_set = VCDataSet(train_data_x, train_data_y, train_length, train_wav_ids, transform=True)
    vcc_dev_data_set = VCDataSet(dev_data_x, dev_data_y, dev_length, dev_wav_ids, transform=True)
    vcc_test_data_set = VCDataSet(test_data_x, test_data_y, test_length, test_wav_ids, transform=True)
    vcc_train_loader = DataLoader(dataset=vcc_train_data_set, batch_size=args.batch_size, shuffle=True)
    vcc_dev_loader = DataLoader(dataset=vcc_dev_data_set, batch_size=args.batch_size, shuffle=False)
    vcc_test_loader = DataLoader(dataset=vcc_test_data_set, batch_size=args.batch_size, shuffle=False)

    print("traing data len: %s \t" % vcc_train_data_set.__len__(), end="")
    print("dev data len: %s \t " % vcc_dev_data_set.__len__(), end="")
    print("test data len: %s" % vcc_test_data_set.__len__())

    rnn = RnnModel(input_size, args.hidden_size, args.num_layers, num_classes, args.drop_out).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(args.num_epochs):
        train_loss = 0.
        for i, tmp in enumerate(vcc_train_loader):
            data = tmp['data']
            label = tmp['label']
            length = tmp['length']
            
            max_len = int(torch.max(length))
            data = data[:, :max_len,:]
            label = label[:, :max_len]
            
            sorted_length, indices = torch.sort(
                length.view(-1), dim=0, descending=True)
            sorted_length = sorted_length.long().numpy()

            data, label = data[indices], label[indices]
            
            data = Variable(data).cuda()

            optimizer.zero_grad()
            outputs, out_length = rnn(data, sorted_length)
            
            outputs = outputs.view(label.size(0), -1)

            loss = criterion(outputs, Variable(label).cuda())

            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]

        print('Epoch [%d/%d] \t Training Loss: %.2f \t' % (epoch + 1, num_epochs,
                                                           train_loss / len(vcc_train_loader)), end="")
        dev_loss = test_loss(rnn, vcc_dev_loader, criterion)
        training_losses.append(train_loss / len(vcc_train_loader))
        dev_losses.append(dev_loss)
        print("Dev Loss: %.2f" % dev_loss)

        scheduler.step(dev_loss)
    print("Min train loss: {} \t Min dev loss: {}".format(min(training_losses), min(dev_losses)))
    final_test_loss = test_loss(rnn, vcc_test_loader, criterion)
    print("Final test loss: {}".format(final_test_loss))
    # save the model
    current_datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    model_name = "./model/{}_{}_{}_{}.pkl".format(current_datetime, args.hidden_size, args.num_layers, args.lr)
    torch.save(rnn, model_name)


if __name__ == '__main__':
    main()









