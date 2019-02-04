from pytorch_src.env_learners.env_learner import EnvLearner
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import time
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PreCoGenModel(nn.Module):
    def __init__(self, state_dim, act_dim, latent_size, drop_rate=0.5):
        super(PreCoGenModel, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h = None

        self.pred_rnn = nn.GRUCell(act_dim, latent_size)
        self.corr_rnn = nn.GRUCell(state_dim, latent_size)

        self.fc1 = nn.Linear(latent_size, latent_size/2)
        self.fc_out = nn.Linear(latent_size/2, state_dim)

    def init_params(self):
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -0.1, 0.1)

    def autoencode(self, x):
        h = self.corr_rnn(x, None)
        out_tmp = self.decode(h)

        # Ensure NOOPs don't get counted towards the loss
        mask = (torch.sum(torch.abs(x), dim=1)>0).float()
        mask = torch.stack([mask]*out_tmp.shape[-1], dim=1)
        out_tmp = out_tmp*mask
        return out_tmp, h

    def decode(self, x):
        out = x
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc_out(out)
        out = torch.tanh(out)
        return  out

    def pred_single(self, a, h):
        h = self.pred_rnn(a, h)
        out_tmp = self.decode(h)

        # Ensure NOOPs don't get counted towards the loss
        mask = (torch.sum(torch.abs(a), dim=1)>0).float()
        mask = torch.stack([mask]*out_tmp.shape[-1], dim=1)
        out_tmp = out_tmp*mask
        return out_tmp, h

    def pred_seq(self, a, h):
        out = torch.zeros(len(a), a[0].shape[0], self.state_dim).to(device)
        for i in range(a.shape[0]):
            out_tmp, h = self.pred_single(a[i], h)
            out[i] = out_tmp
        return out, h

    def forward(self, x, a, y):
        obs = torch.transpose(x, 0, 1)
        act = torch.transpose(a, 0, 1)
        new_obs = torch.transpose(y, 0, 1)

        corr_out, corr_h = self.autoencode(obs[0])
        single_out, single_h = self.pred_single(act[0], corr_h)
        seq_out, seq_h = self.pred_seq(act[1:], single_h)

        corr = torch.mean(torch.abs(corr_out-obs[0]))
        single = torch.mean(torch.abs(single_out-new_obs[0]))
        seq = torch.mean(torch.abs(seq_out-new_obs[1:]))+single

        return corr, single, seq

class PrecoGenEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)

        self.latent_size = 1024
        lr = 1e-5
        self.is_reset = False
        self.max_seq_len = 10
        self.batch_size = 256
        self.model = PreCoGenModel(self.state_dim, self.act_dim, self.latent_size)
        self.model.to(device)
        # self.model.init_params()
        # self.corr_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.single_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.seq_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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

        corr, single, seq = self.get_loss(valid)
        print('Valid Single: ' + str(single))
        print('Valid Seq: ' + str(seq))
        print('Valid Corr: ' + str(corr))
        for i in range(total_steps):
            if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                self.seq_len += 1
                seq_i += 1

            if i % log_interval == 0 and i > 0 and valid is not None:
                corr, single, seq = self.get_loss(valid)
                print('Valid Single: ' + str(single))
                print('Valid Seq: ' + str(seq))
                print('Valid Corr: ' + str(corr))
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
            corr, single, seq = self.train_epoch(train)
            duration = time.time() - start
            if stop_count > early_stopping and early_stopping > 0:
                break
            # if i % log_interval != 0 or i == 0:
            print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's                                          ')
            print('Train Single: ' + str(single))
            print('Train Seq: ' + str(seq))
            print('Train Corr: ' + str(corr))
            print('')
        if valid is not None:
            corr, single, seq = self.get_loss(valid)
            print('Final Epoch')
            print('Valid Single: ' + str(single))
            print('Valid Seq: ' + str(seq))
            print('Valid Corr: ' + str(corr))
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
        if batch_size is None: batch_size = self.batch_size

        Xs = []
        Ys = []
        As = []

        x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        a = deque([np.zeros(self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        y = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)

        for i in range(len(data)):
            obs = data[i][0]/self.state_mul_const
            act = data[i][1]/self.act_mul_const
            new_obs = data[i][3]/self.state_mul_const

            x.appendleft(obs)
            a.appendleft(act)
            y.appendleft(new_obs)

            Xs.append(np.array(x))
            As.append(np.array(a))
            Ys.append(np.array(y))

            reset = data[i][4]

            if reset:
                x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                a = deque([np.zeros(self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                y = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)

        assert len(Ys) == len(As) == len(Xs)

        # p = np.random.permutation(len(X))
        p = np.arange(len(Xs))
        Xs = self.__batch__(np.array(Xs)[p], self.batch_size)
        As = self.__batch__(np.array(As)[p], self.batch_size)
        Ys = self.__batch__(np.array(Ys)[p], self.batch_size)
        return Xs, As, Ys

    def get_loss(self, data):
        Xs, As, Ys = self.__prep_data__(data)
        Corr = 0
        Single = 0
        Seq = 0
        self.model.eval()
        for Xsi, Asi, Ysi \
                in zip(enumerate(Xs), enumerate(As), enumerate(Ys)):
            # data = data.to(device)
            Ysi = torch.from_numpy(Ysi[1].astype(np.float32)).to(device)
            Asi = torch.from_numpy(Asi[1].astype(np.float32)).to(device)
            Xsi = torch.from_numpy(Xsi[1].astype(np.float32)).to(device)

            corr, single, seq = self.model(Xsi, Asi, Ysi)
            Corr += corr.item()
            Single += single.item()
            Seq += seq.item()
        return Corr/len(Xs), Single/len(Xs), Seq/len(Xs)

    def train_epoch(self, data, eager=True):
        import sys, math
        start = time.time()
        self.model.train()
        Xs, As, Ys = self.__prep_data__(data)
        Corr = 0
        Single = 0
        Seq = 0
        idx = 0
        # self.corr_optimizer.zero_grad()
        # self.single_optimizer.zero_grad()
        # self.seq_optimizer.zero_grad()
        for Xsi, Asi, Ysi \
                in zip(enumerate(Xs), enumerate(As), enumerate(Ys)):
            # data = data.to(device)
            Xsi = torch.from_numpy(Xsi[1].astype(np.float32)).to(device)
            Asi = torch.from_numpy(Asi[1].astype(np.float32)).to(device)
            Ysi = torch.from_numpy(Ysi[1].astype(np.float32)).to(device)

            corr, single, seq = self.model(Xsi, Asi, Ysi)
            Corr += corr.item()
            Single += single.item()
            Seq += seq.item()

            seq.backward(retain_graph=True)
            single.backward(retain_graph=True)
            corr.backward(retain_graph=True)

            # self.seq_optimizer.step()
            # self.single_optimizer.step()
            # self.corr_optimizer.step()
            self.optimizer.step()

            idx += 1
            sys.stdout.write(str(round(float(100*idx)/float(len(data)/self.batch_size), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        return Corr/len(Xs), Single/len(Xs), Seq/len(Xs)

    def reset(self, obs_in):
        x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(device)
        self.h = self.model.corr_rnn(x, None)
        self.is_reset = True

    def step(self, action_in, obs_in=None, episode_step=None, save=True, buff=None, num=None):
        self.model.eval()
        if obs_in is not None and not self.is_reset:
            x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(device)
            self.h = self.model.corr_rnn(x, None)
        a = torch.from_numpy(np.array([action_in.astype(np.float32)/self.act_mul_const])).to(device)
        self.h = self.model.pred_rnn(a, self.h)
        self.is_reset = False
        new_obs = self.model.decode(self.h)
        new_obs = new_obs[0].detach().cpu().numpy()*self.state_mul_const
        return new_obs
