import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer

class Seq_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='LSTM', bidirectional=False, num_layers=1):
        super(Seq_RNN, self).__init__()
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # Nếu dùng mạng 2 chiều (bidirectional), rnn_hidden_dim của RNN bằng một nửa hidden_dim 
        # để khi ghép (concatenate) 2 chiều lại sẽ ra đúng kích thước hidden_dim ban đầu.
        rnn_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=rnn_hidden_dim, 
                               num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_hidden_dim, 
                              num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_hidden_dim, 
                              num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f"Không hỗ trợ rnn_type: {rnn_type}")

    def forward(self, forward_seq):
        # forward_seq có shape: (batch_size, seq_len, input_dim)
        out, hn = self.rnn(forward_seq)
        
        # LSTM trả về tuple (h_n, c_n), còn GRU/RNN trả về h_n dạng Tensor
        if isinstance(hn, tuple):
            h_n = hn[0]
        else:
            h_n = hn
            
        # h_n có shape: (num_layers * num_directions, batch_size, rnn_hidden_dim)
        # Lấy trạng thái ẩn ở layer cuối cùng
        num_directions = 2 if self.bidirectional else 1
        last_layer_h = h_n[-num_directions:] # Lấy trạng thái của hướng đi/về của layer cuối
        
        if self.bidirectional:
            # Ghép trạng thái chiều đi (forward) và chiều về (backward) của layer cuối
            c_t = torch.cat((last_layer_h[0], last_layer_h[1]), dim=1) # Shape: (batch_size, hidden_dim)
        else:
            c_t = last_layer_h[0] # Shape: (batch_size, hidden_dim)
            
        return c_t


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)
        self.seq_rnn = Seq_RNN(input_dim=self.num_channels, hidden_dim=configs.TC.hidden_dim, rnn_type='LSTM', bidirectional=True)
        #Modes of choice: LSTM, GRU, RNN

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)
        # c_t = self.seq_rnn(forward_seq)


        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)