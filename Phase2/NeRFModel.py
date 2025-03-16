import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L, pos_encoding):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.pos_encoding = pos_encoding 
        self.pos_L = embed_pos_L
        self.dir_L = embed_direction_L

        if pos_encoding:
            self.LL1 = nn.Linear(60,256)
        else:
            self.LL1 = nn.Linear(3, 256)
        self.relu1 = nn.ReLU()

        self.LL2 = nn.Linear(256,256)
        self.relu2 = nn.ReLU()

        self.LL3 = nn.Linear(256,256)
        self.relu3 = nn.ReLU()

        self.LL4 = nn.Linear(256,256)
        self.relu4 = nn.ReLU()

        self.LL5 = nn.Linear(256,256)
        self.relu5 = nn.ReLU()
        # after relu 5 we concat raw lamda(x) again following paper...
        if pos_encoding:
            self.LL6 = nn.Linear(316,256) # 316 from 256 + 60
        else:
            self.LL6 = nn.Linear(259,256)
        self.relu6 = nn.ReLU()

        self.LL7 = nn.Linear(256,256)
        self.relu7 = nn.ReLU()

        self.LL8 = nn.Linear(256,256)
        self.relu8 = nn.ReLU()

        self.LL9 = nn.Linear(256,256)
        self.LL_Sigma = nn.Linear(256,1)
        # after forward through LL9, we get output sigma and 256 feature vector... 
        # No activation function for either...
        # we then concat lamda(d)
        if pos_encoding:
            self.LL10 = nn.Linear(280,128)
        else:
            self.LL10 = nn.Linear(256, 128)
        self.relu10 = nn.ReLU()

        # one last hidden layer to give us the final 128 feature vector

        self.LL11 = nn.Linear(128,3)
        # output layer after LL11, no ReLU

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

        return x

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################
        PE_pos = self.position_encoding(pos, self.pos_L)
        PE_dir = self.position_encoding(direction, self.pos_L)
        if self.pos_encoding:
            output = self.relu1(self.LL1(PE_pos))
        else:
            output = self.relu1(self.LL1(pos))

        output = self.relu2(self.LL2(output))
        output = self.relu3(self.LL3(output))
        output = self.relu4(self.LL4(output))
        output = self.relu5(self.LL5(output))

        if self.pos_encoding:
            output = torch.cat(output, PE_pos)
        else:
            output = torch.cat(output, pos)
        output = self.relu6(self.LL6(output))

        output = self.relu7(self.LL7(output))
        output = self.relu8(self.LL8(output))

        sigma = self.LL_Sigma(output)
        output = self.LL9(output)

        if self.pos_encoding:
            output = torch.cat(output, PE_dir)
        else:
            output = torch.cat(output, direction)
        output = self.relu10(self.LL10(output))
        output = self.LL11(output)

        return sigma, output
