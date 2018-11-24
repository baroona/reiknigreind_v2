# import agent_double as agent_double
# import agent_onehot as agent_one
# import agent_one_tseFeat as agent_one_tseFeat
import Backgammon
import numpy as np
from tqdm import tqdm
import pickle
# from agent_dyna import NeuralNet

import agent_dyna



# train

# n = 10
# net = agent.policy_nn()
# val_nn = agent.val_func_nn()
# agent = agent_one.net()
# agent = agent_double.net()
# agent = agent_one_tseFeat.net()
# agent = pickle.load(open('saved_net_one', 'rb'))
# agent = pickle.load(open('saved_net_one_2', 'rb'))
# agent = NeuralNet(input_size, hidden_size, num_classes).to(device)
agent = agent_dyna.agent()
# agent.actor.theta = pickle.load(open('saved_net_one', 'rb'))
# print(agent.actor.theta)
train = True
if(train):
    agent.actor.theta = pickle.load(open('saved_net_one', 'rb'))
def main():
    # print(torch.randn(3, 5, requires_grad=True))
    # print(torch.randint(5, (3,), dtype=torch.int64))
    ranges = 1
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0  # Collecting stats of the games
    nGames = 100   # how many games?
    arr = np.zeros(nGames)
    for g in tqdm(range(nGames)):
        # ##Zero eligibility traces (according to psudo code)
        winner = Backgammon.play_a_game(commentary=False, net=agent, train=train)
        winners[str(winner)] += 1
        arr[g] = winner             
        if(g % 10 == 0):

            print(agent.actor.theta)
            k = winners["1"]
            print("winrate is %f" % (k / (g + 0.00000001)))
    # print(winners)
    #  Save the agent
    if(train is True):
        file_net = open('saved_net_one', 'wb')
        pickle.dump(agent.actor.theta, file_net)
        file_net.close()
    print("Out of", ranges, nGames, "games,")
    print("player", 1, "won", winners["1"], "times and")
    print("player", -1, "won", winners["-1"], "times")


if __name__ == '__main__':
    main()
