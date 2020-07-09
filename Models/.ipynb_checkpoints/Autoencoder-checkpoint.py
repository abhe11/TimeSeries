import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import Variable



class Autoencoder(nn.Module):
    
    def __init__(self, n_features, 
                 hidden_size=16,
                 embed_dim=8,
                fc_dim=8):
    """
    """
    
        super(Autoencoder, self).__init__()

        self.hidden_size = hidden_size
        self.embed_dim   = embed_dim
        self.n_features  = n_features
        self.fc_dim      = fc_dim
        self.n_layers    = 1
        
        self.lstm_AE1 = nn.LSTM(n_features,
                               self.hidden_size,
                               num_layers = self.n_layers,
                               batch_first=True)
        self.lstm_AE2 = nn.LSTM(self.hidden_size,
                                self.embed_dim,
                                num_layers = self.n_layers,
                                batch_first=True)
        
        self.dense_1  = nn.Linear(self.fc_dim,
                                 self.embed_dim)
        
        self.lstm_DE1 = nn.LSTM(self.embed_dim,
                                self.embed_dim,
                                num_layers = self.n_layers,
                                batch_first=True)
        self.lstm_DE2 = nn.LSTM(self.embed_dim,
                                self.fc_dim,
                                num_layers = self.n_layers,
                                batch_first=True)
        
        self.loss_fn  = nn.MSELoss()
        
        
        
    def init_hidden(self, batch_size, dim):
        """
        """
        
        hidden_state = Variable(torch.zeros(self.n_layers, 
                                            batch_size, 
                                            dim))
        cell_state = Variable(torch.zeros(self.n_layers, 
                                          batch_size, 
                                          dim))
        return hidden_state, cell_state
        
        
        
    def forward(self, x_encoder, x_decoder):
        """
        """
        
        batch_size, seq_len,_ = x_encoder.size()
        
        hidden_1 = self.init_hidden(batch_size, self.hidden_size)
        e_output_1, _ = self.lstm_AE1(x_encoder, hidden_1)
        e_output_1 = F.dropout(e_output_1, p=0.1, training=True)
        
        hidden_2 = self.init_hidden(batch_size, self.embed_dim)
        e_output_2, (h2,c2) = self.lstm_AE2(e_output_1, hidden_2)
        e_output_2 = F.dropout(e_output_2, p=0.5, training=True)
        
        dense_output = self.dense_1(x_decoder)
        
        hidden_3 = (h2, c2)
        de_output_1, _ = self.lstm_DE1(dense_output, hidden_3)
        de_output_1 = F.dropout(de_output_1, p=0.1, training=True)
        
        hidden_4 = self.init_hidden(batch_size, self.fc_dim)
        de_output_2, _ = self.lstm_DE2(de_output_1, hidden_4)
        de_output_2 = F.dropout(de_output_2, p=0.5, training=True)
        
        return [e_output_2, de_output_2]

    
    
    def loss(self, pred, truth):
        """
        """
        return self.loss_fn(pred, truth)
    
    
    