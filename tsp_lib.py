import math
import numpy as np
import time
from time import sleep
import os
from scipy.spatial import Delaunay

from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


"""
default hyper-parameter configuration:
ebd_size = 128, hidden_size = 128, n_glimpses = 1, beta = 0.9, max_grad_norm = 2.
other 5 different hyper-parameter configurations:
1. ebd_size=64
2. hidden_size=256
3. beta=0.8
4. max_grad_norm = 5.
5. n_glimpses = 5
"""


class DatasetGenerator(Dataset):

    """
        Generating dataset for TSP TaskGenerating dataset for TSP Task
    """

    def __init__(self, num_nodes, num_samples, name='dataset', random_seed=111):
        super(DatasetGenerator, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for _ in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)
        sleep(2)
        print("Successfully generated %s dataset! Dataset size: %d" % (name, num_samples))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]


def reward(sample_solution, use_cuda=True, name='reward'):
    """
    Args:
        sample_solution: n * [batch_size, 2]
    """
    if 'TSP' in name:
        batch_size = sample_solution[0].size(0)
        n = len(sample_solution)
        tour_len = Variable(torch.zeros([batch_size]))

        if use_cuda:
            tour_len = tour_len.cuda()

        for i in range(n - 1):
            distance = torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)
            tour_len += distance

        distance = torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)
        tour_len += distance
        reward_ = tour_len

    return reward_


class Attention(nn.Module):

    """
        Using two types of attention mechanism: "Dot" and "Bahdanau"
    """

    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=True):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, ebd_size, use_cuda=True, use_sdne=True, add_noise=False, is_training=True):
        super(GraphEmbedding, self).__init__()
        self.use_cuda = use_cuda
        self.use_sdne = use_sdne
        self.add_noise = add_noise
        self.is_training = is_training

        ebd_size_1 = ebd_size * 4 if use_sdne else ebd_size
        ebd_size_2 = ebd_size * 2
        ebd_size_3 = ebd_size

        # embed using a 2-D tensor as a weight matrix
        # nn.Parameter makes the tensor become a trainable parameter in the model
        # self.embedding_1 = [2, ebd_size_1]
        self.embedding_1 = nn.Parameter(torch.FloatTensor(input_size, ebd_size_1))
        if self.use_sdne:
            # self.embedding_2 = [ebd_size_1, ebd_size_2]
            self.embedding_2 = nn.Parameter(torch.FloatTensor(ebd_size_1, ebd_size_2))
            # self.embedding_3 = [ebd_size_2, ebd_size_3]
            self.embedding_3 = nn.Parameter(torch.FloatTensor(ebd_size_2, ebd_size_3))
        # initialization
        self.embedding_1.data.uniform_(-(1. / math.sqrt(ebd_size_1)), 1. / math.sqrt(ebd_size_1))
        if self.use_sdne:
            self.embedding_2.data.uniform_(-(1. / math.sqrt(ebd_size_2)), 1. / math.sqrt(ebd_size_2))
            self.embedding_3.data.uniform_(-(1. / math.sqrt(ebd_size_3)), 1. / math.sqrt(ebd_size_3))

    def forward(self, inputs):
        """
        :param inputs: tensor [batch, 2, seq_len]
        :return: embedded: tensor [batch, seq_len, embedding_size]
            Embed each node in the graph to a 128-dimension space
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        # embedding_1 = [batch, 2, ebd_size_1]
        embedding_1 = self.embedding_1.repeat(batch_size, 1, 1)
        if self.use_sdne:
            # embedding_2 = [batch, ebd_size_1, ebd_size_2]
            embedding_2 = self.embedding_2.repeat(batch_size, 1, 1)
            # embedding_3 = [batch, ebd_size_2, ebd_size_3]
            embedding_3 = self.embedding_3.repeat(batch_size, 1, 1)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            # embedding = [batch, 1, ebd_size_1]
            embedding = torch.bmm(inputs[:, :, :, i].float(), embedding_1.cuda())
            if self.use_sdne:
                # embedding = [batch, 1, ebd_size_2]
                embedding = torch.bmm(embedding.float(), embedding_2.cuda())
                # embedding = [batch, 1, ebd_size_3]
                embedding = torch.bmm(embedding.float(), embedding_3.cuda())
            embedded.append(embedding)
        embedded = torch.cat(tuple(embedded), 1)
        return embedded


class PointerNet(nn.Module):
    def __init__(self,
                 ebd_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 attention,
                 use_cuda=True,
                 use_sden=True):
        super(PointerNet, self).__init__()

        self.embedding_size = ebd_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.use_cuda = use_cuda

        self.embedding = GraphEmbedding(2, ebd_size, use_cuda=use_cuda, use_sdne=use_sden)
        # input of nn.LSTM (batch_first=True): (batch_size, seq_len, feature_num)
        self.encoder = nn.LSTM(self.embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention, use_cuda=use_cuda)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(ebd_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(ebd_size)), 1. / math.sqrt(ebd_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def gaussian(self, inputs, is_training, add_noise, mean=0.5, stddev=1.0):
        # print(is_training, add_noise)
        if not is_training and add_noise:
            noise = Variable(inputs.data.new(inputs.size()).normal_(mean, stddev))
            return inputs + noise
        return inputs

    def forward(self, inputs, is_training, add_noise):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        embedded = self.gaussian(embedded, is_training, add_noise)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()

        idxs = None

        # decoder_input = [batch, embedding_size]
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        # for CH and DT, this for loop should be replaced by while loop
        for i in range(seq_len):

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2),).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)

            # torch.multinomial: sampling with multinomial distribution
            # torch.squeeze: eliminate the dimension of size 1    e.g. (4, 1, 3) -> (4, 3)
            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print('RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            # decoder_input = [batch, embedding_size]
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)

        return prev_probs, prev_idxs


class CombinatorialRL(nn.Module):
    """
        Optimization with policy gradients

        Model-free policy-based Reinforcement Learning to optimize the parameters of a pointer network
        using the well-known REINFORCE algorithm
    """
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 reward,
                 attention,
                 use_cuda=True,
                 use_sdne=True,
                 name='model'):
        super(CombinatorialRL, self).__init__()
        start_time = time.time()
        self.reward = reward
        self.use_cuda = use_cuda
        self.name = name

        self.actor = PointerNet(
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            use_cuda,
            use_sdne)

        print("\nSuccessfully built %s model!" % name)
        print("Time cost for building %s model: %4.4f s" % (name, (time.time()-start_time)))

    def forward(self, inputs, is_training, add_noise):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """

        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        seq_len = inputs.size(2)

        probs, action_idxs = self.actor(inputs, is_training=is_training, add_noise=add_noise)

        actions = []
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

        action_probs = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        R = self.reward(actions, self.use_cuda, self.name)

        return R, action_probs, actions, action_idxs


class TrainModel:
    """
        Simple Training Class
    """
    def __init__(self, args, model, train_dataset, val_dataset, batch_size=128, threshold=None,
                 continue_train=False, max_grad_norm=2., name='model'):
        self.model = model
        self.name = name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.save_dir = "./save"
        self.img_dir = "./imgs"
        self.continue_train = continue_train

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.actor_optim = optim.Adam(model.actor.parameters(), lr=1e-4)
        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.critic_exp = []
        self.advantage = []
        self.train_loss = []
        self.val_tour = []

        self.start_epoch = 0

        if self.continue_train:
            # load entire model
            # there must be only one checkpoint
            load_dir = os.path.join(self.save_dir, self.name,
                                         "train_dataset_size_%d" % len(self.train_dataset),
                                         "val_dataset_size_%d" % len(self.val_dataset),
                                         "n_glimpses_%d" % args.n_glimpses,
                                         "ebd_size_%d_hidden_size_%d" % (args.ebd_size, args.hidden_size),
                                         "beta_%.2f_max_grad_norm_%.2f" % (args.beta, args.max_grad_norm),
                                         "use_sdne_%s_%d" % ("True" if args.use_sdne else "False", args.ebd_size))
            if args.add_noise:
                load_dir = os.path.join(load_dir, "add_noise")
            self.load_model(load_dir)

        self.epochs = self.start_epoch

        print("\nModel's state_dict:")
        # Print model's state_dict
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size(), "\t",
                  self.model.state_dict()[param_tensor].get_device())

    def train_and_validate(self, args, n_epochs):
        critic_exp_mvg_avg = torch.zeros(1)
        if args.use_cuda:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        start_time = time.time()
        print("\nTraining starts!")

        for epoch in range(self.start_epoch, n_epochs):
            data_loaders = enumerate(self.train_loader)
            for batch_id, sample_batch in data_loaders:
                self.model.train()

                inputs = Variable(sample_batch)
                inputs = inputs.cuda()

                R, probs, actions, actions_idxs = self.model(inputs, is_training=True, add_noise=args.add_noise)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                elif args.use_decay and epoch < 5:
                    # the critic's expectation decays with training steps
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * args.beta) + ((1. - args.beta) * R.mean())
                elif args.use_decay and epoch >= 5:
                    critic_exp_mvg_avg = critic_exp_mvg_avg * args.gamma
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * args.beta) + ((1. - args.beta) * R.mean())

                # compute the difference between critics' expectation and R
                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                # minimize actor loss
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                              float(self.max_grad_norm), norm_type=2)

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().detach())
                self.critic_exp.append(critic_exp_mvg_avg.detach())
                self.advantage.append(advantage.mean().detach())
                self.train_loss.append(actor_loss.detach())

                if batch_id % 100 == 0:
                    self.plot(args, self.epochs, batch_id)
                    print("Epoch [%d/%d] Batch [%d] Elapsed time: %4.4f s" % (epoch, (n_epochs-1), batch_id,
                                                                                 time.time()-start_time))
                    self.model.eval()
                    for val_batch in self.val_loader:
                        inputs = Variable(val_batch)
                        inputs = inputs.cuda()

                        R, probs, actions, actions_idxs = self.model(inputs, is_training=False, add_noise=args.add_noise)
                        self.val_tour.append(R.mean().item())

            # save entire model at the end of an epoch
            save_dir = os.path.join(self.save_dir, self.name,
                                    "train_dataset_size_%d" % len(self.train_dataset),
                                    "val_dataset_size_%d" % len(self.val_dataset),
                                    "n_glimpses_%d" % args.n_glimpses,
                                    "ebd_size_%d_hidden_size_%d" % (args.ebd_size, args.hidden_size),
                                    "beta_%.2f_max_grad_norm_%.2f" % (args.beta, args.max_grad_norm),
                                    "use_sdne_%s_%d" % ("True" if args.use_sdne else "False", args.ebd_size))
            if args.add_noise:
                save_dir = os.path.join(save_dir, "add_noise_2")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_name = "Neural_Combinatorial_Optimization_%d" % self.epochs
            self.save_model(save_dir, model_name)

            if self.threshold and self.train_tour[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break

            self.epochs += 1

        print("\nTraining Completes!",
              "Total time cost: %4.2f s" % (time.time()-start_time),
              "Final training reward: %4.4f" % self.train_tour[-1],
              "Final validation reward: %4.4f" % self.val_tour[-1])

    def plot(self, args, epoch, batch_id):
        if 'TSP' in args.problem_name:
            clear_output(True)
            plt.figure(figsize=(12, 12))
            plt.ion()
            plt.subplot(221)
            plt.title('train tour length: epoch %d batch_id %d reward %s' %
                      (epoch, batch_id, '%4.2f' % self.train_tour[-1] if len(self.train_tour) else 'collecting'))
            plt.plot(self.train_tour, color='blue', linestyle='-')
            plt.plot(self.critic_exp, color='pink', linestyle='-')
            plt.grid()
            plt.subplot(222)
            plt.title('val tour length: epoch %d batch_id %d reward %s' %
                      (epoch, batch_id, '%4.2f' % self.val_tour[-1] if len(self.val_tour) else 'collecting'))
            plt.plot(self.val_tour)
            plt.grid()
            plt.subplot(223)
            plt.title('train loss: epoch %d batch_id %d loss %s' %
                      (epoch, batch_id, '%4.2f' % self.train_loss[-1] if len(self.train_loss) else 'collecting'))
            plt.plot(self.train_loss, color='coral', linestyle='-')
            plt.grid()
            plt.subplot(224)
            plt.title('advantage: epoch %d batch_id %d advantage %s' %
                      (epoch, batch_id, '%4.2f' % self.advantage[-1] if len(self.advantage) else 'collecting'))
            plt.plot(self.advantage, color='purple', linestyle='-')
            plt.grid()
            img_dir = "%s_train_dataset_size_%d_val_dataset_size_%d" % (self.name, len(self.train_dataset),
                                                                        len(self.val_dataset),)
            save_dir = os.path.join(self.img_dir, img_dir, 'n_glimpses_%d' % args.n_glimpses,
                                    "ebd_size_%d_hidden_size_%d" % (args.ebd_size, args.hidden_size),
                                    "beta_%.2f_max_grad_norm_%.2f" % (args.beta, args.max_grad_norm),
                                    ("use_sdne_%d" % args.ebd_size if args.use_sdne else 'use_sdne_False'))
            if args.add_noise:
                save_dir = os.path.join(save_dir, "add_noise_2")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_name = "epoch_%d_batch_%d.jpg" % (epoch, batch_id)
            plt.savefig(os.path.join(save_dir, img_name))
            plt.close()

    def save_model(self, save_dir, model_name):
        print("Epoch %d: Saving model..." % self.epochs)
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.actor_optim.state_dict(),
            'train_tour': self.train_tour,
            'critic_exp': self.critic_exp,
            'advantage': self.advantage,
            'train_loss': self.train_loss,
            'val_tour': self.val_tour
        }, os.path.join(save_dir, model_name+'.tar'))
        with open(os.path.join(save_dir, "checkpoint.txt"), 'w') as f:
            f.write(model_name+'.tar')
            f.close()
        print("Save SUCCESS!")

    def load_model(self, save_dir):
        print("Loading model...")
        checkpoint_log_path = os.path.join(save_dir, 'checkpoint.txt')
        if os.path.exists(checkpoint_log_path):
            with open(checkpoint_log_path, 'r') as f:
                file_name = f.readline()
                f.close()
            checkpoint = torch.load(os.path.join(save_dir, file_name))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.train_tour = checkpoint['train_tour']
            self.critic_exp = checkpoint['critic_exp']
            self.advantage = checkpoint['advantage']
            self.train_loss = checkpoint['train_loss']
            self.val_tour = checkpoint['val_tour']
            print("Load SUCCESS!")
        else:
            print("Load failed...")
