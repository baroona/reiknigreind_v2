#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import flipped_agent
import numpy as np
import Backgammon
from numpy.random import choice


def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return abs(x) * (x > 0)
# hyperparameters

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    # e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)




class agent():
    def __init__(self):
        self.actor = actor()
        self.search = search()
        # self.theta = np.random(198)
        # A = []
        # B = []



class search():
    def __init__(self):
        self.theta = np.zeros(198)
        self.z = np.zeros(198)
        self.alpha = 0.01
        self.lamb = 0.9


    def getValue(self, board, actor_theta, player):
        x = np.dot(getFeatures(board, player), actor_theta) + np.dot(getFeatures(board, player), np.transpose(self.theta))
        return tanh(x)

    def nextMove(self, board, dice, player, actor_theta):
        possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
        
        if(len(possible_moves) == 0):
            return [], []
        board_vals = np.zeros(len(possible_boards))
        for k in range(0, len(possible_boards)):
            board_vals[k] = self.getValue(possible_boards[k], actor_theta, player)

        pi_vals = softmax(board_vals)
        index = np.arange(0, len(possible_boards))
        i = choice(index, p=pi_vals)
        move = possible_moves[i]
        newBoard = possible_boards[i]

        return move, newBoard

    def do(self, board_real, dice, actor_theta, player):
        commentary = False
        print_results = False
        for i in range(0, 2):
            board = np.copy(board_real)
            old_state = np.copy(board_real)
            self.z = np.zeros(198)
            if(len(board) == 0):
                break
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
                        if(count < 2):

                            a, s_prime = self.nextMove(board_copy, dice, player, actor_theta)
                            
                            new_move = a
                            if(len(s_prime) != 0):
                                old_state = s_prime
                        else:
                            new_move, new_state = self.nextMove(board_copy, dice, player, actor_theta)
                        move = new_move
                        
                        
                    elif player == -1:
                        move = Backgammon.random_agent(board_copy, dice, player, i)
                    if len(move) != 0:

                        for m in move:
                            board = Backgammon.update_board(board, m, player)
                        if(player == 1 and count > 1):
                            new_state = np.copy(board)  
                            if(not Backgammon.game_over(new_state) and not Backgammon.check_for_error(new_state)):


                                delta = 0 + self.getValue(new_state, actor_theta, player) - self.getValue(old_state, actor_theta, player)
                                self.theta = self.theta + (self.alpha * delta * self.z)
                                self.z = self.lamb * self.z + getFeatures(old_state, player)
                                old_state = new_state

                    if commentary:
                        print("Simulationgame: move from player", player, ":")
                        Backgammon.pretty_print(board)
                player = -player
                count = count + 1   
            if(print_results):
                print("simulation game nr", i)
                Backgammon.pretty_print(board)
            delta = player * -1 + 0 - self.getValue(old_state, actor_theta, player)
            
            self.theta = np.add(self.theta , (self.alpha * delta * self.z))
            self.z = self.lamb * self.z + getFeatures(old_state, player)
              
            
            
            
class actor():
    def __init__(self):
        self.theta = np.zeros(198)
        # self.theta = [-3.18999441,-1.03887380,9.33341845,5.30173598
        #                 ,-2.09768145,-1.04925168,-1.25749622,6.89753760
        #                 ,-2.53754539,3.74207343,5.78762120,5.09694394
        #                 ,2.31441518,2.25321355,-9.72708283,-5.84377536
        #                 ,-6.74949896,1.61515787,1.44267788,5.71951928
        #                 ,-2.37155935,3.88789619,4.48013632,-1.09307591
        #                 ,1.57561549,9.53299779,-2.06461400,-2.71979860
        #                 ,-2.89128391,5.91696587,1.01812444,1.07084738
        #                 ,-3.17586268,-7.12016896,-5.56258731,-2.24686471
        #                 ,-4.50994833,-2.68416192,-5.80302652,-6.67545255
        #                 ,-3.03699933,-7.45317543,1.55338860,0.00000000
        #                 ,-3.64634962,-1.00331775,7.28537274,0.00000000
        #                 ,-4.61921180,-2.55105265,-2.81470853,1.24861160
        #                 ,-4.25645858,-1.48516595,3.18233631,3.84672213
        #                 ,-5.71976686,-2.42075141,-5.27419754,-1.18178390
        #                 ,-3.69171868,-8.91546524,-1.68992109,3.14493730
        #                 ,-4.23937471,-2.16897585,-7.10345102,-7.31999965
        #                 ,-3.07220318,-5.08127456,-1.00151716,-9.71010493
        #                 ,-8.32015526,-3.67144005,-1.49051156,-2.85225223
        #                 ,-7.64064354,-5.10359074,-2.82030595,-2.05260487
        #                 ,-6.94443814,-4.20548656,-2.24477163,-5.48479555
        #                 ,2.19269870,-6.49298202,-1.03664646,-3.32620159
        #                 ,-3.70361217,-4.77249368,-1.79006079,-3.84877796
        #                 ,0.00000000,0.00000000,0.00000000,0.00000000
        #                 ,5.74414201,7.86369753,5.11350107,1.49432008
        #                 ,9.07299376,2.18373868,1.16726152,8.17672263
        #                 ,7.52083388,6.85172074,3.91775059,1.11615638
        #                 ,5.13246704,2.58963876,1.12525965,-3.41563198
        #                 ,4.94338164,2.59582017,2.20240695,1.05937041
        #                 ,8.11623089,4.18349172,1.76605356,6.62032948
        #                 ,3.15971443,-1.45733460,5.71701958,2.53592784
        #                 ,5.29639110,1.96199399,5.51152856,2.39307688
        #                 ,3.19299672,1.49457291,8.84504461,-1.05415312
        #                 ,1.66297390,1.17126993,7.05926552,9.76572395
        #                 ,1.51188896,1.09716419,5.56766949,2.33126545
        #                 ,8.69275970,6.02773627,3.62423533,1.01201963
        #                 ,5.71923220,-3.42942495,-2.73185645,0.00000000
        #                 ,5.41297021,1.46840492,1.85920241,-2.05841370e-12
        #                 ,4.73474234,1.43397447,4.52513360,1.95171527
        #                 ,2.45020209,-2.41715148,5.03661834,1.41751420
        #                 ,3.62868436,2.14259889,9.45255365,-1.10153315
        #                 ,2.69921791,4.64549283,4.69844435,-2.67672214
        #                 ,1.60654818,1.40101152,1.37160359,6.37207519
        #                 ,5.65952321,1.13313716,-6.40788878,-2.19452044
        #                 ,4.42252917,2.01469131,1.25200828,9.71862747
        #                 ,-6.10794990,-1.13705309,-1.09559217,-1.48648601
        #                 ,-3.11399351,-5.61818777,-6.30313597,-4.30081312
        #                 ,0.00000000,0.00000000,0.00000000,0.00000000
        #                 ,-6.24907045,-16.43228805,5.02262909,6.01355009
        #                 ,-1.66990715,-2.29171581]
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
            board_vals[k] = self.getValue(possible_boards[k], search_theta, player)

        # pi_vals = softmax(board_vals)
        # index = np.arange(0, len(possible_boards))
        # i = choice(index, p=pi_vals)
        # print(board_vals.argmax())
        # i = list(board_vals).index(board_vals.argmax())
        # i = np.where(max(board_vals) == board_vals)
        #if(len())
        i = board_vals.argmax()
        # print("board_vals")
        # print(board_vals)
        # print("list i ")
        # print(i)
        # i = choice(i)
        # print(i)
        move = possible_moves[i]  # ##Pick the next move according to the index selected
        newBoard = possible_boards[i]  # ##Pick the nex board according to the index selected
        return move, newBoard



    def getValue(self, board, search_theta, player):
        x = np.dot(getFeatures(board, player), self.theta) + np.dot(getFeatures(board, player), search_theta)
        return tanh(x)


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
    


    if player == -1:
        move = flipped_agent.flip_move(move)  # ##Flip the move
    return move
