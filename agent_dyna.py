#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import flipped_agent
import torch
import torch.nn as nn
import numpy as np
import Backgammon
from torch.autograd import Variable
from numpy.random import choice


# hyperparameters

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




class agent():
    def __init__(self):
        self.actor = actor()
        self.search = search()
        # self.theta = np.random(198)
        self.gamma = 0.8
        self.delta = 0
        # A = []
        # B = []



class search():
    def __init__(self):
        self.theta = np.zeros(198)
        self.z = np.zeros(198)
        self.alpha = 0.0001
        self.lamb = 0.7


    def getValue(self, board, actor_theta):
        # s = getFeatures(board, 1)
        x = np.dot(board, actor_theta) + np.dot(board, np.transpose(self.theta))
        return x

    def nextMove(self, board, dice, player, actor_theta):
        possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player=1)
        
        if(len(possible_moves) == 0):
            return [], []
        # feature_boards = []
        board_vals = np.zeros(len(possible_boards))
        for k in range(0, len(possible_boards)):
            # feature_boards.append()
            board_vals[k] = self.getValue(getFeatures(possible_boards[k], player), actor_theta)

        pi_vals = softmax(board_vals)
        index = np.arange(0, len(possible_boards))
        i = choice(index, p=pi_vals)
        move = possible_moves[i]
        newBoard = possible_boards[i]

        return move, newBoard

    def do(self, board_real, dice, actor_theta, player):
        commentary = False
        print_results = False
        for i in range(0, 10):
            # print(i)
            # print("started do loop") 
            # print("search theta in sim loop")
            # print(self.theta)
            # print("z up top")
            # print(self.z) 
            # s = np.copy(board)
            board = np.copy(board_real)
            old_state = np.copy(board_real)
            self.z = np.zeros(198)
            # a, s_prime = self.nextMove(np.copy(board), dice, player, actor_theta)
            # check_first_move = 0
            # check_no_move = 0
            if(len(board) == 0):
                break
            # s_no_move = s
            count = 0
            while not Backgammon.game_over(board) and not Backgammon.check_for_error(board):
                if commentary:
                    print("Simulationgame: lets go player ", player)

                dice = Backgammon.roll_dice()
                if commentary:
                    print("Simulationgame: rolled dices:", dice)

                # make a move (2 moves if the same number appears on the dice)
                for i in range(1 + int(dice[0] == dice[1])):
                    board_copy = np.copy(board)
                    if player == 1:
                        # print("playermove")
                        # if(check_first_move == 1):
                            # for m in a:
                            #     s_prime = Backgammon.update_board(s, m, 1)
                        if(count < 2):
                            # print("first search started")

                            # net.search.do(board_copy, dice, net.actor.theta, player)
                            a, s_prime = self.nextMove(board_copy, dice, player, actor_theta)
                            
                            new_move = a
                            if(len(s_prime) != 0):
                                old_state = s_prime
                            # move = self.a
                        # elif(check_no_move == 1):
                        #    a, s_prime = self.nextMove(board_copy, dice, player, actor_theta)
                        #     check_no_move = 0
                            
                        #    new_move = a
                            # new_state = self.s_prime
                        else:
                            new_move, new_state = self.nextMove(board_copy, dice, player, actor_theta)
                            # old_move = net.actor.a
                            # old_state = net.actor.s_prime
                        move = new_move
                        # if(len(s_prime) != 0 and not Backgammon.game_over(s_prime) and not Backgammon.check_for_error(s_prime)):
                        #     a_prime, s_prime_prime = self.nextMove(s_prime, dice, player, actor_theta)
                        #     if(len(a_prime) == 0):
                        #         check_no_move = 1
                        #         s_no_move = s
                        #     else:
                        #         delta = 0 + sigmoid(self.getValue(getFeatures(s_prime_prime, player), actor_theta)) - sigmoid(self.getValue(getFeatures(s_prime, player), actor_theta))
                        #         # print("DELTA INNER LOOP")
                        #         # print(delta)
                        #         # self.theta = self.theta + self.alpha * delta * self.z
                        #         # print(self.theta)
                        #         self.theta = self.theta + (self.alpha * delta * self.z)
                        #         self.z = self.lamb * self.z + getFeatures(s_prime, player)
                                
                        #         a = a_prime
                        
                        # s = s_prime
                        
                    elif player == -1:
                        # print("randommove")
                        move = Backgammon.random_agent(board_copy, dice, player, i)
                    

                    if len(move) != 0:
                        
                        # print(move)
                        for m in move:
                            board = Backgammon.update_board(board, m, player)
                        if(player == 1 and count > 1):
                            new_state = np.copy(board)  
                            if(not Backgammon.game_over(new_state) and not Backgammon.check_for_error(new_state)):

                                # a_prime, s_prime_prime = self.nextMove(board, dice, player, actor_theta)
                                #if(len(a_prime) == 0 or len(s_prime_prime) == 0):
                                #    check_no_move = 1
                                #else:
                                delta = 0 + sigmoid(self.getValue(getFeatures(new_state, player), actor_theta)) - sigmoid(self.getValue(getFeatures(old_state, player), actor_theta))
                                self.theta = self.theta + (self.alpha * delta * self.z)
                                self.z = self.lamb * self.z + getFeatures(old_state, player)
                                # a = a_prime
                                old_state = new_state
                            # else:
                            #    break

                    if commentary:
                        print("Simulationgame: move from player", player, ":")
                        Backgammon.pretty_print(board)
                # check_first_move = 1
                player = -player
                count = count + 1   
            # print("oldstate is", old_state)
            if(print_results):
                print("simulation game nr", i)
                Backgammon.pretty_print(board)
            delta = player * -1 + 0 - sigmoid(self.getValue(getFeatures(old_state, player), actor_theta))
            # if(delta > 0.1):
            #     print("DELTA")
            #     print(delta)
            #     print("theta Update !!!!!")
            #     print(np.add(self.theta , (np.multiply(self.alpha * delta, self.z))))
            #     print("z")
            #     print(self.z)
            # print("theta val")
            # print(self.theta) 
            self.theta = np.add(self.theta , (self.alpha * delta * self.z))
            self.z = self.lamb * self.z + getFeatures(old_state, player)
              
            
            
            # print("theta val")
            # print(self.theta)
class actor():
    def __init__(self):
        self.theta = np.zeros(198)
        self.z = np.zeros(198)
        self.alpha = 0.0001
        self.lamb = 1
    

    def nextMove(self, board, dice, player, search_theta):
        possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player=1)
        
        if(len(possible_moves) == 0):
            return [], []
        # feature_boards = []
        board_vals = np.zeros(len(possible_boards))
        for k in range(0, len(possible_boards)):
            # feature_boards.append()
            board_vals[k] = self.getValue(getFeatures(possible_boards[k], player), search_theta)

        pi_vals = softmax(board_vals)
        index = np.arange(0, len(possible_boards))
        i = choice(index, p=pi_vals)
        # a_prime
        move = possible_moves[i]  # ##Pick the next move according to the index selected
        newBoard = possible_boards[i]  # ##Pick the nex board according to the index selected
        return move, newBoard



    def getValue(self, board, search_theta):
        # print("board * actor theta")
        # print(np.dot(board, self.theta))
        # print("board * search theta")
        # print(search_theta)
        x = np.dot(board, self.theta) + np.dot(board, search_theta)
        return x


def getFeatures(board, player):
    features = np.zeros((198))
    for i in range(1, 25):
        try:
            board_val = board[i]
        except:
            print("exception")
            print(board)
            raise 
            return features
            
        # else:
        #    commit()
        
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


def action(net, board_copy, dice, player, i):

    if player == -1:
        board_copy = flipped_agent.flip_board(board_copy)  # #Flip the board
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)

    if len(possible_moves) == 0:
        return []
    move = []
    # if (net.firstMove is True):
    #     net.actor.s = board_copy
    #     net.search.do(board_copy, dice, actor_theta, player)
    #     net.actor.a, net.actor.s_prime = net.actor.nextMove(board, dice, player)
    # # else:
    # #    s_prime = Backgammon.update_board(s, a, 1)
    # R = 0
    # if(Backgammon.game_over(net.actor.s_prime) or Backgammon.check_for_error(net.actor.s_prime)):  # ##Did I win? If so the reward shall be +1
    #     R = 1
    #     target = 0 
    #     delta = R + 0 - getValue(getFeatures(net.actor.s_prime))
    #     z = net.actor.lamb * net.actor.zeta + net.actor + getFeatures(s_prime)
    #     return 
    # net.search.do(net.actor.s_prime)
    # net.actor.a_prime net.actor.s_prime_prime = net.actor.nextMove(net.actor.s_prime, dice, player)
    # delta = R + getValue(getFeatures(net.actor.s_prime_prime)) - getValue(getFeatures(net.actor.s_prime))
    # net.actor.theta = net.actor.theta + net.actor.alpha * delta * net.actor.z
    # net.actor.z = net.actor.lamb * net.actor.z + getFeatures(s_prime)
    # net.actor.s = net.actor.s_prime
    # net.actor.a = net.actor.a_prime


            # print(R + net.gamma * target - oldtarget)
    # feature_boards = []
    # board_vals = np.zeros(len(possible_moves))
    # #  ##Create new features using Tsesauros
    # for k in range(0, len(possible_boards)):
    #     feature_boards.append(getFeatures(possible_boards[k], player))
    #     board_vals[k] = net.critic.getValue(feature_boards[k])

    # # CHOOSE MOVE a
    # j = board_vals.tolist().index(max(board_vals))
    # move = possible_moves[j]

    # # EXECUTE A OBSERVE R AND S'
    # newBoard = possible_boards[j]
    # newBoardFeatures = getFeatures(newBoard, player)


    # # SEARCH S'
    

    # # ## Critic feedforward
    # target, oldtarget = net.critic.forward(newBoardFeatures, getFeatures(board_copy, player))  # (newBoardFeatures,getFeatures(board_copy,player) )

    # R = 0
    # if(Backgammon.game_over(newBoard) or Backgammon.check_for_error(newBoard)):  # ##Did I win? If so the reward shall be +1
    #     R = 1
    #     target = 0  # # #Terminal state is 0
    #     # print(R + net.gamma * target - oldtarget)


    # delta = R + net.gamma * target - oldtarget
    # # ##Update the critic via backpropgation
    # net.critic.backward(R, delta, net.gamma)


    if player == -1:
        move = flipped_agent.flip_move(move)  # ##Flip the move
    return move
