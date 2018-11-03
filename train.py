import agent
import Backgammon
import numpy as np
# train

# n = 10
# net = agent.policy_nn()
# val_nn = agent.val_func_nn()
new_agent = agent.net()


def main():
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0  # Collecting stats of the games
    nGames = 1000  # how many games?
    arr = np.zeros(nGames)
    for g in range(nGames):
        winner = Backgammon.play_a_game(commentary=False, net=new_agent)
        winners[str(winner)] += 1
        arr[g] = winner
        if(g % 100 == 0):
            print(new_agent.torch_nn_policy.theta)
    # print(winners)
    file = open('Failed.py', 'w')
    file.write(np.array_str(arr))
    file.close()
    print("Out of", nGames, "games,")
    print("player", 1, "won", winners["1"], "times and")
    print("player", -1, "won", winners["-1"], "times")


if __name__ == '__main__':
    main()
