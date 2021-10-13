import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import f1_score
from learner import Learner, Scaling, Translation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Meta(nn.Module):
    def __init__(self, args, config, config_scal, config_trans, feat, label_num, adj_tilde, adj):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.h = feat

        self.MI = True  # use params initialization
        self.hidden = feat.shape[1]  # the second dim feature
        self.mlp = MLP(self.hidden)
        fc_params = nn.Linear(self.hidden, self.n_way, bias=None)
        self.fc = [fc_params.weight.detach()] * self.task_num
        for i in range(self.task_num): self.fc[i].requires_grad = True

        self.net = Learner(config)
        self.net = self.net.to(device)

        self.scaling = Scaling(config_scal, args, label_num)
        self.scaling = self.scaling.to(device)

        self.translation = Translation(config_trans, args, label_num)
        self.translation = self.translation.to(device)

        #self.embeddings = Encoder(args.hop, feat.shape[1]) # useless
        #self.g.ndata['h'] = self.embeddings(feat, adj_tilde, adj) # useless

        self.meta_optim = optim.Adam([{'params':self.net.parameters()}, {'params':self.mlp.trans.parameters()},
                                      {'params':self.scaling.parameters()}, {'params':self.translation.parameters()}], lr=self.meta_lr)
        #self.embeddings_optim = optim.Adam(self.embeddings.parameters(), lr=args.lr, weight_decay=args.weight_decay) # useless

    def reset_fc(self):
        self.fc = [torch.Tensor(self.n_way, self.hidden)]*self.task_num

    def prework(self, meta_information):
        return self.mlp(meta_information)

    def preforward(self, support, fc):
        return F.linear(support, fc, bias=None)

    def forward(self, x_spt, y_spt, x_qry, y_qry, meta_information_dict, training):
        """
        b: number of tasks
        setsz: the size for each task
        :param x_spt:   {task_num : id_support}, where each unit is a support set, i.e. x_spt[0] is a np.array()
        :param y_spt:   {task_num : id_support_label}, where each unit is a support label, i.e. y_spt[0] is a corresponding torch.Tensor()
        :param x_qry:   same as x_spt
        :param y_qry:   same as y_spt
        :param meta_information_dict: every class prototype embedding
        :return:
        """
        # task_num = len(x_spt)
        #meta_information = id_by_class_prototype_embedding
        step = self.update_step if training is True else self.update_step_test
        querysz = self.n_way * self.k_qry
        losses_s = [0 for _ in range(step)]
        losses_q = [0 for _ in range(step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(step + 1)]
        f1s = [0 for _ in range(step + 1)]

        for i in range(self.task_num):
            meta_information = meta_information_dict[i] # [n_way, features.size(1)]
            self.fc[i] = self.prework(meta_information) # [n_way, features.size(1)]
            logits_two = self.preforward(self.h[x_spt[i]], self.fc[i]) # the meta information of x_support
            logits_three = self.preforward(self.h[x_qry[i]], self.fc[i]) # the meta information of x_query

            # 1. run the i-th task and compute loss for k=0
            logits_value = self.net(logits_two, vars=None)#[x_spt[i]] # logits_value is intermediate variable
            '''
            # task-agnostic
            logits_value_soft = F.softmax(logits_value, dim=-1) # this may be useless
            h_theta = torch.sum(-torch.mul(logits_value_soft, torch.log(logits_value_soft))) # useless
            '''
            scaling = self.scaling(logits_value)
            translation = self.translation(logits_value)
            adapted_prior = []
            for s in range(len(scaling)):
                adapted_prior.append(torch.mul(self.net.parameters()[s], (scaling[s] + 1)) + translation[s])
            logits = self.net(logits_two, adapted_prior)
            # logits = self.net(self.g, self.g.ndata['h'], vars=None)
            # logits = self.net(self.g, self.g.ndata['h'], vars=None)[x_spt[i]]
            # loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
            # task agnostic
            '''
            logits_soft = F.softmax(logits, dim=-1) # useless
            h_theta_update = torch.sum(-torch.mul(logits_soft, torch.log(logits_soft))) # useless
            '''
          
            loss = F.cross_entropy(logits, y_spt[i]) #+ (h_theta_update - h_theta) * 0.001
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, adapted_prior)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapted_prior)))
            #grad = torch.autograd.grad(loss, self.net.parameters())
            #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update

            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(logits_three, adapted_prior)
                '''
                logits_q_soft = F.softmax(logits_q, dim=-1) # useless
                h_theta_q = torch.sum(-torch.mul(logits_q_soft, torch.log(logits_q_soft)) / logits_q.shape[0])  # useless
                '''
                #logits_q = self.net(self.g, logits_three, self.net.parameters())
                #logits_q = self.net(self.g, self.g.ndata['h'], self.net.parameters())[x_qry[i]]
                # loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q
                f1s[0] = f1s[0] + f1_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(logits_three, fast_weights)
                #logits_q = self.net(self.g, self.g.ndata['h'], fast_weights)[x_qry[i]]
                # loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q
                f1s[1] = f1s[1] + f1_q

            for k in range(1, step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(logits_two, fast_weights)
                #logits = self.net(self.g, self.g.ndata['h'], fast_weights)[x_spt[i]]
                # loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                '''
                logits_soft = F.softmax(logits, dim=-1) # useless
                h_theta_update = torch.sum(-torch.mul(logits_soft, torch.log(logits_soft))) # useless
                loss = F.cross_entropy(logits, y_spt[i]) + (h_theta_update - h_theta) * 0.001 # useless
                '''
                loss = F.cross_entropy(logits, y_spt[i])
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # this is modify
                logits_q = self.net(logits_three, fast_weights)
                #logits_q = self.net(self.g, self.g.ndata['h'], fast_weights)[x_qry[i]]
                # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                '''
                logits_q_soft = F.softmax(logits_q, dim=-1) # useless
                h_theta_q_update = torch.sum(-torch.mul(logits_q_soft, torch.log(logits_q_soft)) / logits_q.shape[0])  # useless
                loss_entropy = 1 * (h_theta_q_update - h_theta_q)# useless
                '''
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))

                if training == True:
                    l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                    l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                    l2_loss = l2_loss * 0.0001
                    losses_q[k + 1] += (loss_q + l2_loss)
                else:
                    losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q
                f1s[k + 1] = f1s[k + 1] + f1_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / self.task_num
        if training == True:
            if torch.isnan(loss_q):
                pass
            else:
            # optimize theta parameters
              self.meta_optim.zero_grad()
              #self.embeddings_optim.zero_grad() # useless
              loss_q.backward()
              self.meta_optim.step()
              #self.embeddings_optim.step() # useless

        accs = np.array(corrects) / (self.task_num * querysz)
        f1_sc = np.array(f1s) / (self.task_num)

        return accs, f1_sc


class MLP(nn.Module):
    def __init__(self, n_way):  # n_way = feature.shape[1]
        super(MLP, self).__init__()
        self.way = n_way
        self.hidden = n_way
        self.trans = nn.Linear(self.way, self.hidden)

        self.att_map = nn.Linear(self.way, self.hidden)
        self.att = False


    def forward(self, inputs): # inputs:[n_way, features.size(0)]
        params = self.trans(inputs)
        params = F.normalize(params, dim=-1)

        if self.att:
            att = self.att_map(inputs)
            att = F.normalize(att, dim=-1)
            return params, att
        else:
            return params


class Encoder(nn.Module):
    def __init__(self, hop, nfeat):
        super(Encoder, self).__init__()
        self.hop = hop
        #alpha = nn.Parameter(torch.ones(2 * self.hop + 1))
        #self.alpha = F.softmax(alpha, dim=0)
        self.linear = nn.Linear(nfeat * (2 * self.hop + 1), nfeat)

    def forward(self, fea, adj_tilde, adj):
        list_mat = [fea]
        list_cat = []
        X = fea
        X_tilde = fea
        for i in range(self.hop):
            X = torch.spmm(adj, X)
            X_tilde = torch.spmm(adj_tilde, X_tilde)
            list_mat.append(X)
            list_mat.append(X_tilde)
        for i in range(self.hop * 2 + 1):
            X_f = list_mat[i]
            #X_f = self.HopNorm(X_f)
            list_cat.append(torch.mul(X_f, self.alpha[i]))
        H = torch.cat(list_cat, dim=1)
        Z = F.relu(self.linear(H))
        return Z

    def HopNorm(self, X_f):
        return torch.div(X_f, torch.norm(X_f, p=2, dim=1, keepdim=True))



