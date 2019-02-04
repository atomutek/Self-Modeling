from pytorch_src.env_learners.env_learner import EnvLearner
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import time
from collections import deque

from torch.autograd import Variable

class DenseRNN:
    def __init__(self, input, hidden_size, n_layers, rnn_cell=nn.LSTMCell):
        self.res = []
        self.layers = []
        in_size = input
        if type(hidden_size) == list: out_size = hidden_size[0]
        else: out_size = hidden_size

        # Appends a rnn_cell for each of the sizes in hidden_size if a list or all at hidden_size if an int
        for i in range(n_layers):
            if type(hidden_size) == list: out_size = hidden_size[i]
            layer = rnn_cell(in_size, out_size)
            self.layers.append(layer)
            in_size = out_size

        # Goes through each of the layers and adds the intermediate dense connections as rnn_cells
        for i in range(n_layers-1):
            layer_res = []
            for j in range(i+1, n_layers):
                layer = rnn_cell(self.layers[i].input_size, self.layers[j].hidden_size)
                layer_res.append(layer)
            self.res.append(layer_res)

        self.device = type(self.layers[0].bias_hh.device)

    def to(self, device):
        if self.layers[0].bias_hh.device == device: return
        for layer in self.layers:
            layer = layer.to(device)
        for list in self.res:
            for layer in list:
                layer = layer.to(device)
        return self

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.parameters())
        for list in self.res:
            for layer in list:
                params.append(layer.parameters())
        return params

    def cuda(self):
        for layer in self.layers:
            layer = layer.cuda()
        for list in self.res:
            for layer in list:
                layer = layer.cuda()
        return self

    # Properly initializes all the hidden vectors of the rnn_cells with 0 vectors
    def init_hidden(self, x):
        hidden_layer = []
        hidden_res = []
        for layer in self.layers:
            hidden = (torch.zeros(x.shape[0], layer.hidden_size), torch.zeros(x.shape[0], layer.hidden_size))
            if x.is_cuda: hidden = (hidden[0].cuda(), hidden[1].cuda())
            hidden_layer.append(hidden)
        for list in self.res:
            hidden_res_list = []
            for layer in list:
                hidden = (torch.zeros(x.shape[0], layer.hidden_size), torch.zeros(x.shape[0], layer.hidden_size))
                if x.is_cuda: hidden = (hidden[0].cuda(), hidden[1].cuda())
                hidden_res_list.append(hidden)
            hidden_res.append(hidden_res_list)
        return (hidden_layer, hidden_res)

    def __call__(self, data, h=None):
        if h is None: h = self.init_hidden(data)
        out = data
        out_res = []
        for i in range(len(self.layers)):
            h[0][i] = self.layers[i](out, h[0][i])
            new_out = h[0][i][0]
            for j in range(len(out_res)):
                res = out_res[j].pop(0)
                new_out += res

            if i < len(self.res):
                layer_res = []
                for j in range(len(self.res[i])):
                    h[1][i][j] = self.res[i][j](out, h[1][i][j])
                    layer_res.append(h[1][i][j][0])
                out_res.append(layer_res)

            out = new_out

        return out, h

class Gen(nn.Module):
    def __init__(self, input_size, latent_size, state_dim, hidden_size, nb_layers):
        super(Gen, self).__init__()
        self.rnn = DenseRNN(input_size, hidden_size, nb_layers).to(device)
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.Linear(latent_size, state_dim),
        )

class SeqModel(nn.Module):
    def __init__(self, input_size, latent_size, state_dim, act_dim, nb_layers = 3):
        super(SeqModel, self).__init__()

        self.latent_size = latent_size
        self.state_dim = state_dim
        self.training = True
        hidden_size = latent_size
        # hidden_size = [latent_size*4, latent_size*2, latent_size]
        self.gen = Gen(input_size, latent_size, state_dim, hidden_size, nb_layers)
        self.disc_fc = nn.Sequential(
            nn.Linear(input_size+state_dim, 4*latent_size),
            nn.Linear(4*latent_size, 2*latent_size),
            nn.Linear(2*latent_size, 2*latent_size),
            nn.Linear(2*latent_size, latent_size),
            nn.Linear(latent_size, state_dim),
        )

    def discriminator(self, SAS):
        return torch.sigmoid(self.disc_fc(SAS))

    def generate(self, x, h=None):
        inputs = torch.transpose(x, 0, 1)
        out = torch.zeros(inputs.shape[0], inputs.shape[1], self.state_dim).to(device)
        for i in range(inputs.shape[0]):
            out_tmp, h = self.gen.rnn(inputs[i], h)
            out_tmp = torch.tanh(self.gen.fc(out_tmp))+inputs[i][:,:self.state_dim]
            out[i] += out_tmp
        return torch.transpose(out, 0, 1), h

    def forward(self, data, y):
        close_lambda = 0.0

        G, h = self.generate(data)
        SA = torch.transpose(data, 0, 1)
        Sg = torch.transpose(G, 0, 1)
        Sx = torch.transpose(y, 0, 1)
        SASg = torch.cat([SA, Sg], -1)
        SASx = torch.cat([SA, Sx], -1)
        DG = self.discriminator(SASg)
        DX = self.discriminator(SASx)

        mse = torch.mean(torch.abs(G-y))

        disc = torch.mean(torch.log(DX)+torch.log(1-DG))
        gen = torch.mean(torch.log(1-DG))+close_lambda*mse

        return gen, disc, mse

def kl_loss(mean, var):
    return -0.5 * torch.mean(1 + var - mean**2 - var.exp())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SeqEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)

        self.batch_size = 128
        lr = 1e-5
        self.max_seq_len = 60
        self.latent_dim = 32
        nb_layers = 1


        self.model = SeqModel(self.act_dim+self.state_dim, self.latent_dim, self.state_dim, self.act_dim, nb_layers)
        # self.model = VRNN(self.act_dim+2*self.state_dim, self.latent_dim, self.state_dim, self.act_dim)
        self.model.to(device)
        self.disc_optimizer = optim.Adam(self.model.disc_fc.parameters(), lr=lr)
        self.gen_optimizer = optim.Adam(self.model.gen.parameters(), lr=lr)

    def initialize(self, session, load=False):
        return NotImplementedError

    def train(self, train, total_steps, valid=None, log_interval=1, early_stopping=-1, save_str=None, verbose=True):
        min_loss = 10000000000
        stop_count = 0

        seq_i = 0
        seq_idx = [1] * (self.max_seq_len - self.seq_len + 1)
        for j in range(1, self.max_seq_len - self.seq_len + 1):
            seq_tmp = self.max_seq_len - j
            seq_idx[j] = (seq_tmp + 1) * seq_idx[j - 1] / seq_tmp
        seq_idx.reverse()
        mul_const = total_steps / sum(seq_idx)
        for j in range(len(seq_idx)):
            seq_idx[j] = round(mul_const * seq_idx[j])
            if j > 0:
                seq_idx[j] += seq_idx[j - 1]

        gen, disc, mse = self.get_loss(valid)
        print('Valid Gen: ' + str(gen))
        print('Valid Disc: ' + str(disc))
        print('Valid MSE: ' + str(mse))
        for i in range(total_steps):
            if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                self.seq_len += 1
                seq_i += 1

            if i % log_interval == 0 and i > 0 and valid is not None:
                gen, disc, mse = self.get_loss(valid)
                print('Valid Gen: ' + str(gen))
                print('Valid Disc: ' + str(disc))
                print('Valid MSE: ' + str(mse))
                print('')
                if save_str is not None:
                    torch.save(self.model, 'models/'+str(save_str))
                    # self.model = torch.load('models/'+str(save_str))
                    # gen_test, disc_test, mse_test = self.get_loss(valid)
                    # assert gen_test == gen
                    # assert disc_test == disc
                    # assert mse_test == mse
                    print("Model saved in path: %s" % 'models/'+save_str)
            start = time.time()
            gen, disc, mse = self.train_epoch(train)
            duration = time.time() - start
            if stop_count > early_stopping and early_stopping > 0:
                break
            # if i % log_interval != 0 or i == 0:
            print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's                                          ')
            print('Train Gen: ' + str(gen))
            print('Train Disc: ' + str(disc))
            print('Train MSE: ' + str(mse))
            print('')
        if valid is not None:
            gen, disc, mse = self.get_loss(valid)
            print('Final Epoch')
            print('Valid Gen: ' + str(gen))
            print('Valid Disc: ' + str(disc))
            print('Valid MSE: ' + str(mse))
            print('')
            if save_str is not None:
                torch.save(self.model, 'models/'+str(save_str))
                # self.model = torch.load('models/'+str(save_str))
                # gen_test, disc_test, mse_test = self.get_loss(valid)
                # assert gen_test == gen
                # assert disc_test == disc
                # assert mse_test == mse
                print("Model saved in path: %s" % 'models/'+save_str)

    def __prep_data__(self, data, batch_size=None):
        X = []
        reset=True
        if batch_size is None: batch_size = self.batch_size

        S = []
        Xs = []
        Ys = []
        As = []

        s = deque([np.zeros(self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        y = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        a = deque([np.zeros(self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)

        for i in range(len(data)):
            # obs = data[i][0]/self.state_mul_const
            # act = data[i][1]/self.act_mul_const
            # new_obs = data[i][3]/self.state_mul_const
            obs = data[i][0]/self.state_mul_const
            act = data[i][1]/self.act_mul_const
            new_obs = data[i][3]/self.state_mul_const

            if reset:
                s = deque([np.zeros(self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                # s.append([np.concatenate([obs, act, new_obs])])
            # else:
            x.appendleft(obs)
            a.appendleft(act)
            y.appendleft(new_obs)
            s.appendleft(np.concatenate([obs, act]))

            X.append(obs)
            As.append(a)
            Ys.append(y)
            Xs.append(np.array(x))
            S.append(np.array(s))

            reset = data[i][4]

        assert len(X) == len(Ys) == len(As) == len(S) == len(Xs)

        # p = np.random.permutation(len(X))
        # p = np.arange(len(X))
        # X = self.__batch__(np.array(X)[p], self.batch_size)
        # Y = self.__batch__(np.array(Y)[p], self.batch_size)
        # A = self.__batch__(np.array(A)[p], self.batch_size)
        # S = self.__batch__(np.array(S)[p], self.batch_size)
        # Xs = self.__batch__(np.array(Xs)[p], self.batch_size)

        X = DataLoader(TensorDataset(torch.Tensor(X).to(device)), batch_size=batch_size)
        Ys = DataLoader(TensorDataset(torch.Tensor(Ys).to(device)), batch_size=batch_size)
        As = DataLoader(TensorDataset(torch.Tensor(As).to(device)), batch_size=batch_size)
        S = DataLoader(TensorDataset(torch.Tensor(S).to(device)), batch_size=batch_size)
        Xs = DataLoader(TensorDataset(torch.Tensor(Xs).to(device)), batch_size=batch_size)

        return X, Ys, As, S, Xs

    def get_loss(self, data):
        X, Ys, As, S, Xs = self.__prep_data__(data)
        Disc = 0
        Gen = 0
        MSE = 0
        for Xi, Ysi, Asi, Si, Xsi \
                in zip(enumerate(X), enumerate(Ys), enumerate(As), enumerate(S), enumerate(Xs)):
            # data = data.to(device)
            Ysi = Ysi[1][0]
            Asi = Asi[1][0]
            Si = Si[1][0]
            Xsi = Xsi[1][0]

            gen, disc, mse = self.model(Si, Ysi)
            Gen += gen.item()
            Disc += disc.item()
            MSE += mse.item()
        return Gen/len(X), Disc/len(X), MSE/len(X)

    def train_epoch(self, data, eager=True):
        import sys, math
        start = time.time()
        self.model.train()
        X, Ys, As, S, Xs = self.__prep_data__(data)
        sys.stdout.write('Loaded Data                                                                                                                                                \r')
        Disc = 0
        Gen = 0
        MSE = 0
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        # for Xi, Ysi, Asi, Si, Xsi \
        #         in zip(enumerate(X), enumerate(Ys), enumerate(As), enumerate(S), enumerate(Xs)):
        #     # data = data.to(device)
        #     Ysi = Ysi[1][0]
        #     Asi = Asi[1][0]
        #     Si = Si[1][0]
        #     Xsi = Xsi[1][0]
        #
        #     gen, disc, mse = self.model(Si, Ysi)
        #     # MSE += mse.item()
        #
        #     # if not math.isnan(gen):
        #     #     Gen += gen.item()
        #     #     gen.backward(retain_graph=True)
        #     if not math.isnan(disc) and disc.item() > -10:
        #         Disc += disc.item()
        #         disc.backward(retain_graph=True)
        #
        #     # mse.backward(retain_graph=True)
        #
        #     # self.gen_optimizer.step()
        #     self.disc_optimizer.step()
        #     sys.stdout.write(str(round(float(50*Xi[0])/float(len(data)/self.batch_size), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        for Xi, Ysi, Asi, Si, Xsi \
                in zip(enumerate(X), enumerate(Ys), enumerate(As), enumerate(S), enumerate(Xs)):
            # data = data.to(device)
            Ysi = Ysi[1][0]
            Asi = Asi[1][0]
            Si = Si[1][0]
            Xsi = Xsi[1][0]

            gen, disc, mse = self.model(Si, Ysi)
            # Gen += gen.item()
            MSE += mse.item()

            # if not math.isnan(gen) and gen.item() > -10:
            #     Gen += gen.item()
            #     MSE += mse.item()
            #     gen.backward(retain_graph=True)
            # if not math.isnan(disc):
            #     Disc += disc.item()
            #     disc.backward(retain_graph=True)

            mse.backward(retain_graph=True)

            self.gen_optimizer.step()
            # self.disc_optimizer.step()
            sys.stdout.write(str(50+round(float(50*Xi[0])/float(len(data)/self.batch_size), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        return Gen/len(X), Disc/len(X), MSE/len(X)

    def reset(self, obs_in):
        return NotImplementedError

    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        return NotImplementedError

    def next_move(self, obs_in, episode_step):
        return np.random.uniform(-1, 1, self.act_dim)
