import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import rationale_net.models.mlp as mlp
import rationale_net.utils.learn as learn
import pdb

class Generator(nn.Module):

    def __init__(self, n_features, args):
        super(Generator, self).__init__()
        self.args = args
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time = False)
        else:
            self.mlp = mlp.MLP(args)

        self.z_dim = 2

        self.hidden = nn.Linear(1, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)

    def z_forward(self, x):
        '''
        Returns prob of each token being selected
        '''
        logits = self.hidden(x)
        probs = learn.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        z = probs[:,:,1] # shape = [B, N_FT]
        return z

    def forward(self, x_indx):
        '''
        Given input x_indx of dim (batch, length), return z (batch, length) such that z
        can act as element-wise mask on x
        '''
        if self.args.model_form == 'cnn':
            #x = self.embedding_layer(x_indx.squeeze(1))
            x = x_indx
            if self.args.cuda:
                x = x.cuda()
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x)
        else:
            x = x_indx
            if self.args.cuda:
                x = x.cuda()
            #x = torch.transpose(x, 1, 2)
            #activ = self.mlp(x)
            #x = torch.transpose(x_indx, 1, 2)
            #activ = self.mlp(x_indx)
        
        activ = x
        z = self.z_forward(F.relu(activ))
        mask = self.sample(z)
        return mask, z


    def sample(self, z):
        '''
        Get mask from probablites at each token. Use gumbel
        softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            ## pointwise set <.5 to 0 >=.5 to 1
            mask = learn.get_hard_mask(z)
        return mask


    def loss(self, mask, x_indx):
        '''
        Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        selection_cost = torch.mean(torch.sum(mask, dim=0)) # torch.mean(torch.sum(mask, dim=1))
        #l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        #r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        #continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return selection_cost #, continuity_cost
