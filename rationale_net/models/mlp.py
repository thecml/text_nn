import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class MLP(nn.Module):
	def __init__(self, args):
		super(MLP, self).__init__() #load super class for training data
		self.fc1 = nn.Linear(14, args.hidden_dim) #defining fully connected with input 784 and output 320
		self.relu = nn.ReLU()
  
	def forward(self, x): #feed forward
		layer1 = x.view(-1, 14) #make it flat in one dimension from 0 - 784
		return self.relu(self.fc1(layer1)) #layer2 = layer1 -> fc1 -> relu



	