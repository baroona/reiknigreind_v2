#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon
import random
import math
import torch
from torch.autograd import Variable
import pickle

# import copy


def sigmoid_derivative(x):
    return x * (1.0 - x)


def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))


def getValue(board, player):
    features = getFeatures(board, player)
    num_hidden_units = 41
    num_output_units = 1
    num_inputs = 198
    hl_bias = 1
    ol_bias = 1
    input_weights = [[np.random.randn() for x in range(num_hidden_units)] for y in range(num_inputs)]
        # 40 x 2
    hidden_weights = [[np.random.randn() for x in range(num_output_units)] for y in range(num_hidden_units)]
    hl = sigmoid(np.matmul(features, input_weights) + hl_bias)
    ol = sigmoid(np.matmul(np.transpose(hl), hidden_weights) + ol_bias)
    return ol


def getValue2(theta, board, player):
    features = getFeatures(board, player)
    iw = np.reshape(theta[0:7920], (198, 40))
    hw = np.reshape(theta[7920:7960], (40, 1))
    hl = sigmoid(np.matmul(features, iw))
    ol = sigmoid(np.matmul(np.transpose(hl), hw))
    return ol


def getFeatures(board, player):
    features = np.zeros((198))
    for i in range(1, 24):
        board_val = board[i]
        place = (i - 1) * 4
        if(board_val < 0):
            place = place + 96
        # if(board_val == 0):
        #     features[place:place + 4] = 0
        if(abs(board_val) == 1):
            # print("one in place %i", place)
            features[place] = 1
            features[place + 1:place + 4] = 0
        if(abs(board_val) == 2):
            # print("two in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 0
            features[place + 3] = 0
        if(abs(board_val) == 3):
            # print("three in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 1
            features[place + 3] = 0
        if(abs(board_val) > 3):
            # print("more than three in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 1
            features[place + 3] = ((abs(board_val) - 3) / 2)
    features[192] = board[25] / 2
    features[193] = board[26] / 2
    features[194] = board[27] / 15
    features[195] = board[28] / 15
    if(player == 1):
        features[196] = 1
        features[197] = 0
    else:
        features[196] = 0
        features[197] = 1
    return features


class net():
    def __init__(self):
        # self.val_func_nn = val_func_nn()
        # self.policy_nn = policy_nn()
        self.torch_nn = torch_nn()
        self.torch_nn_policy = torch_nn_policy()
        self.i = 1
        self.gamma = 1


class torch_nn():

    def __init__(self):
        self.device = torch.device('cpu')
        self.w1 = Variable(torch.randn(40, 198, device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.b1 = Variable(torch.zeros((40, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.w2 = Variable(torch.randn(1, 40, device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.b2 = Variable(torch.zeros((1, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.y_sigmoid = 0
        self.target = 0
        self.alpha = 0.001

    def forward(self, x):
        # x = Variable(torch.tensor(one_hot_encoding(board, player), dtype = torch.float, device = device)).view(2*9,1)
        # now do a forward pass to evaluate the new board's after-state value
        x_prime = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(198, 1)
        # x_prime = torch.tensor(x, dtype=torch.float, device=self.device)
        h = torch.mm(self.w1, x_prime) + self.b1  # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid()  # squash this with a sigmoid function
        y = torch.mm(self.w2, h_sigmoid) + self.b2  # multiply with the output weights w2 and add bias
        self.y_sigmoid = y.sigmoid()  # squash this with a sigmoid function
        self.target = self.y_sigmoid.detach().cpu().numpy()
        # lets also do a forward past for the old board, this is the state we will update
        # h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
        # h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        # y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        # y_sigmoid = y.sigmoid() # squash the output
        # delta2 = 0 + gamma * target - y_sigmoid.detach().cpu().numpy() # this is the usual TD error
        return self.target

    def backward(self, gamma, delta):
        self.y_sigmoid.backward()
        # update the eligibility traces using the gradients
        # zero the gradients
        # delta2 = 0 + gamma * self.target - self.y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # perform now the update for the weights
        delta = torch.tensor(delta, dtype=torch.float, device=self.device)
        self.w1.data = self.w1.data + self.alpha * delta * self.w1.grad.data
        self.b1.data = self.b1.data + self.alpha * delta * self.b1.grad.data
        self.w2.data = self.w2.data + self.alpha * delta * self.w2.grad.data
        self.b2.data = self.b2.data + self.alpha * delta * self.b2.grad.data
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()


class torch_nn_policy():

    def __init__(self):
        self.device = torch.device('cpu')
        self.w1 = Variable(torch.randn(40, 23, device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.b1 = Variable(torch.zeros((40, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.w2 = Variable(torch.randn(1, 40, device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.b2 = Variable(torch.zeros((1, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.y_sigmoid = 0
        self.target = 0
        self.alpha = 0.001
        self.theta = np.random.random_sample(23)
        self.alpha_theta = 0.01

    def forward(self, x):
        # x = Variable(torch.tensor(one_hot_encoding(board, player), dtype = torch.float, device = device)).view(2*9,1)
        # now do a forward pass to evaluate the new board's after-state value
        x_prime = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(23, 1)
        # x_prime = torch.tensor(x, dtype=torch.float, device=self.device)
        h = torch.mm(self.w1, x_prime) + self.b1  # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid()  # squash this with a sigmoid function
        y = torch.mm(self.w2, h_sigmoid) + self.b2  # multiply with the output weights w2 and add bias
        self.y_sigmoid = y.sigmoid()  # squash this with a sigmoid function
        self.target = self.y_sigmoid.detach().cpu().numpy()
        # lets also do a forward past for the old board, this is the state we will update
        # h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
        # h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        # y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        # y_sigmoid = y.sigmoid() # squash the output
        # delta2 = 0 + gamma * target - y_sigmoid.detach().cpu().numpy() # this is the usual TD error
        # self.backward(0.5)
        return self.target

    def backward(self, gamma):
        self.y_sigmoid.backward()
        # update the eligibility traces using the gradients
        # zero the gradients
        delta = 0 + gamma * self.target - self.y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # perform now the update for the weights
        delta2 = torch.tensor(delta, dtype=torch.float, device=self.device)
        self.w1.data = self.w1.data + self.alpha * delta2 * self.w1.grad.data
        self.b1.data = self.b1.data + self.alpha * delta2 * self.b1.grad.data
        self.w2.data = self.w2.data + self.alpha * delta2 * self.w2.grad.data
        self.b2.data = self.b2.data + self.alpha * delta2 * self.b2.grad.data
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()


def epsilon_nn_greedy(board, player, epsilon, w1, b1, w2, b2, debug=False):
    moves = Backgammon.legal_moves(board)
    if np.random.uniform() < epsilon:
        if debug is True:
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        # encode the board to create the input

        # FEATURES eru X

        # va[i] = y.sigmoid()
    return moves[np.argmax(va)]


def softmax(possible_moves, possible_boards, board, player, net):
    n = len(possible_moves)
    # print(" nr of possibl moves is %i", len(possible_moves))
    store = []
    # store2 = []
    pol = np.zeros(n)
    pol2 = np.zeros(n)
    s = 0
    # print(possible_moves)
    for i in range(0, n):
        # temp_val = np.dot(np.transpose(net.policy_nn.theta), net.policy_nn.feed_forward(possible_boards[i][1:24]))
        # print("this is temp_val %i", temp_val)
        # temp_val = np.dot(np.transpose(net.policy_nn.theta), possible_boards[i][1:24])
        store.append([possible_boards[i], possible_moves[i]])
        # print(net.torch_nn_policy.forward(possible_boards[i][1:24]))
        # pol[i] = round(math.exp(np.dot(net.torch_nn_policy.theta, net.torch_nn_policy.forward(possible_boards[i][1:24]))), 7)
        pol[i] = round(math.exp(np.dot(net.torch_nn_policy.theta, (possible_boards[i][1:24]))), 12)

        # net.torch_nn_policy.backward(0.5)
        s = s + pol[i]
    # s = np.sum(pol)
    # print("this is pol")
    # print(pol)
    val = np.zeros(23)
    for j in range(0, n):
        pol2[j] = pol[j] / (s + 0.00000000000001)
        val = val + (pol2[j] * possible_boards[j][1:24])

    val = board[1:24] - val
    # print("pol")
    # print(pol)
    # print("pol2")
    # print(pol2)
    # print(len(pol))
    # print("pol2")
    # print(len(pol2))
    # print("store")
    # print(len(store))
    # print("listthing1")
    # print(random.choices(pol, pol2, k=100))
    # print("listthing2")
    # print(random.sample(random.choices(pol, pol2, k=100), 1))
    # print(random.choices(pol, pol2, k=100))
    if(len(pol) == 1):
        return store[0], val
    return store[pol.tolist().index(random.sample(random.choices(pol, pol2, k=100), 1)[0])], val


def action(board_copy, dice, player, i, net=None):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy

    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)

    # console.log(possible_moves)
    # if there are no moves available
    if len(possible_moves) == 0:
        return []

    gamma = 0.5

    ret_arr, softmax_deriv = softmax(possible_moves, possible_boards, board_copy, player, net)
    s_prime = ret_arr[0]

    # print(0 + gamma * net.torch_nn.forward(getFeatures(s_prime, player)) - net.torch_nn.forward(getFeatures(board_copy, player)))

    # delta = 0 + gamma * net.val_func_nn.forward(s_prime, player) - net.val_func_nn.forward(board_copy, player)
    delta = 0 + gamma * net.torch_nn.forward(getFeatures(s_prime, player)) - net.torch_nn.forward(getFeatures(board_copy, player))
    # print(delta)
    net.torch_nn.backward(gamma, delta)
    # print("delta is %i", delta)
    # net.val_func_nn.w = net.val_func_nn.w + (net.val_func_nn.alpha_w * delta * net.val_func_nn.backward(board_copy, player))
    # net.policy_nn.theta = np.append(np.ravel(nn.input_weights), nn.hidden_weights)
    # backprop
    # HERA WEIGHTS I SITTHVORU LAGI

    net.torch_nn_policy.theta = net.torch_nn_policy.theta + net.torch_nn_policy.alpha_theta * net.i * delta * softmax_deriv
    net.i = 0.5 * net.i
    # if(i > 1)

    return ret_arr[1]