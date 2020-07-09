import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import Variable



class Forecaster():
    
    def __init__(self,
                ip_dim,
                op_dim,
                hidden_size1=128,
                hidden_size2=64):
        """
        """
        super(Forecaster, self).__init__()
        
        self.ip_dim = ip_dim
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.op_dim = op_dim
        self.n_layers = 1
        
        self.lstm_1 = nn.LSTM(self.ip_dim,
                               self.hidden_size1,
                               num_layers = self.n_layers,
                               batch_first=True)
        
        self.dense_2  = nn.Linear(self.hidden_size1,
                                  self.hidden_size2)

        self.dense_3  = nn.Linear(self.hidden_size2,
                                  self.op_dim)
        self.loss_fn  = nn.MSELoss()
        
        


    def init_hidden(self, batch_size, dim):
        """
        """
        
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, dim))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, dim))
        return hidden_state, cell_state
    
    
    
    def forward(self, X):
        """
        """
        
        batch_size, seq_len,_ = X.size()
        
        hidden_1    = self.init_hidden(batch_size, self.hidden_size1)
        output_1, _ = self.lstm_1(X, hidden_1)
        output_1    = F.dropout(output_1, p=0.1, training=True)
                
        output_2 = self.dense_12(output_1)
        output_2 = torch.tanh(output_2)
        output_2 = F.dropout(output_2, p=0.2, training=True)
        
        output_3 = self.dense_12(output_2)
        output_3 = torch.tanh(output_3)
        output_3 = F.dropout(output_3, p=0.3, training=True)

        return output_3
    
    
    
    def loss(self, pred, truth):
        """
        """
        return self.loss_fn(pred, truth)

    
    