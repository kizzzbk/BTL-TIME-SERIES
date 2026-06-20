from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

import torch
from torch import nn

class BaselineRNN(nn.Module):
    def __init__(self, configs, rnn_type='LSTM', hidden_dim=128, num_layers=2, bidirectional=True):
        super(BaselineRNN, self).__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Số kênh đầu vào của dữ liệu chuỗi thời gian (ví dụ: SleepEDF=1, HAR=9)
        self.input_size = configs.input_channels
        
        # Khởi tạo mô hình RNN tương ứng
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f"Không hỗ trợ rnn_type: {rnn_type}")
            
        # Số chiều đầu ra của RNN (nếu là hai chiều thì nhân đôi chiều ẩn)
        self.classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Lớp tuyến tính ánh xạ sang số lượng nhãn phân loại (logits)
        self.logits = nn.Linear(self.classifier_input_dim, configs.num_classes)
        
    def forward(self, x_in):
        # x_in đầu vào có shape: (batch_size, input_channels, seq_len)
        # Chuyển đổi sang: (batch_size, seq_len, input_channels) để đưa vào RNN
        x = x_in.transpose(1, 2)
        
        out, hn = self.rnn(x)
        # out shape: (batch_size, seq_len, classifier_input_dim)
        
        # Trích xuất đặc trưng cuối cùng (lấy trạng thái ẩn hn của layer cuối)
        if isinstance(hn, tuple):
            h_n = hn[0] # Đối với LSTM (hn, cn)
        else:
            h_n = hn    # Đối với GRU và RNN thường
            
        num_directions = 2 if self.bidirectional else 1
        last_layer_h = h_n[-num_directions:] # Lấy trạng thái của các hướng ở layer cuối
        
        if self.bidirectional:
            # Ghép chiều đi (forward) và chiều về (backward)
            h_last = torch.cat((last_layer_h[0], last_layer_h[1]), dim=1) # (batch_size, hidden_dim * 2)
        else:
            h_last = last_layer_h[0] # (batch_size, hidden_dim)
            
        # Tính logits phân loại
        logits = self.logits(h_last)
        
        # Trả về (logits, features) để khớp với cấu trúc mong đợi của trainer.py
        return logits, out

