import argparse
import gym
import numpy as np
import random
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from termcolor import colored, cprint

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
# reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
# done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
# info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.

    # 0 1 2
    # 3   4
    # 5 6 7
def updateDir(X, Y, dir):
    if(dir == 0 or dir == 3 or dir == 5):
        X = X - 1
    if(dir == 2 or dir == 4 or dir == 7):
        X = X + 1
    if(dir == 0 or dir == 1 or dir == 2):
        Y = Y - 1
    if(dir == 5 or dir == 6 or dir == 7):
        Y = Y + 1
    return X, Y

def isAdjacent(X1, Y1, X2, Y2):
    if(X1 == X2 and Y1 == Y2):
        return False
    elif(abs(X1 - X2) > 1 or
         abs(Y1 - Y2) > 1):
        return False
    else:
        return True
        
class SantoriniSpec():
    def __init__(self):
        self.reward_threshold = 93

class Santorini():
    def __init__(self):
        self.state = np.zeros([3,5,5],np.int8)
        self.spec = SantoriniSpec()
        self.buildX = -1
        self.buildY = -1
        self.errX, self.errY = -1, -1

    def reset(self):
        self.state = np.zeros([3,5,5],np.int8)
        self.AX = random.randrange(0,5)
        self.AY = random.randrange(0,5)
        self.BX = random.randrange(0,5)
        self.BY = random.randrange(0,5)
        self.buildX = -1
        self.buildY = -1
        self.errX, self.errY = -1, -1
        return self.observation()

    def observation(self):
        self.P = np.zeros([2,5,5],np.int8)
        self.P[0][self.AX][self.AY] = 1
        self.P[1][self.BX][self.BY] = 1
        return np.append(self.state,self.P, axis=0)

    def seed(self, val):
        random.seed(val)

    # action is [2][5][5].  [0] -> build, [1] = moveA
    # 0 1 2
    # 3   4
    # 5 6 7
    def step(self, action):
        self.buildX, self.buildY = -1, -1
        self.errX, self.errY = -1, -1
        moveDir, buildDir = action
        newX, newY = updateDir(self.AX, self.AY, moveDir)
        buildX, buildY = updateDir(newX, newY, buildDir)
        penalty = 0
        # newX, newY = np.argmax(moveA)
        # buildX, buildY = np.argmax(build)

        # Hack to avoid running off the edge.
        newX = newX % 5
        newY = newY % 5
        buildX = buildX % 5
        buildY = buildY % 5

        # Check move is adjacent to current position.
        if(not self.isLegalAction(newX, newY, buildX, buildY)):
            penalty = penalty + 10 
            self.errX = newX       
            self.errY = newY
        else:
            # Update state
            self.AX, self.AY = newX, newY
            self.buildX, self.buildY = buildX, buildY
            if(self.state[0][buildX][buildY] == 0):
                self.state[0][buildX][buildY] = 1
            elif(self.state[1][buildX][buildY] == 0):
                self.state[1][buildX][buildY] = 1
            elif(self.state[2][buildX][buildY] == 0):
                self.state[2][buildX][buildY] = 1

        return self.observation(), self.reward() - penalty, self.done(), []

    def isLegalAction(self, newX, newY, buildX, buildY):
        # Check move is adjacent to current position.
        if(not isAdjacent(newX, newY, self.AX, self.AY)):
            return False
        # Can't move where the opponnent is
        if(newX == self.BX and newY == self.BY):
            return False
        # Can only go up one level.
        if(self.state[2][newX][newY] == 1 and self.state[1][newX][newY] == 0):
            return False
        if(self.state[1][newX][newY] == 1 and self.state[0][newX][newY] == 0):
            return False
        # Check build is adjacent to current position.
        if(not isAdjacent(buildX, buildY, newX, newY)):
            return False
        # Can't build past the top
        if(self.state[2][buildX][buildY] == 1):
            return False
        # Can't build where the opponent is
        if(buildX == self.BX and buildY == self.BY):
            return False
        return True

    def reward(self):
        # First check winning conditions
        x = self.AX
        y = self.AY
        if(self.state[0][x][y] == 1 and
            self.state[1][x][y] == 1 and
            self.state[2][x][y] == 1):
            return 100              
        x = self.BX
        y = self.BY
        if(self.state[0][x][y] == 1 and
            self.state[1][x][y] == 1 and
            self.state[2][x][y] == 1):
            return -100 
        return -1

    def done(self):
        x = self.AX
        y = self.AY
        if(self.state[0][x][y] == 1 and
            self.state[1][x][y] == 1 and
            self.state[2][x][y] == 1):
            return True              
        x = self.BX
        y = self.BY
        if(self.state[0][x][y] == 1 and
            self.state[1][x][y] == 1 and
            self.state[2][x][y] == 1):
            return True
        return False      

    def render(self):
        buildings = self.state[0] + self.state[1] + self.state[2]
        print("---------")
        for y in range(5):
          for x in range(5):
            s = str(buildings[x][y])
            if(x == self.errX and y == self.errY):
                cprint(s, "white", "on_red", end =" ")            
            elif(x == self.AX and y == self.AY):
                cprint(s, "white", "on_green", end =" ")
            elif(x == self.BX and y == self.BY):
                cprint(s, "white", "on_blue", end =" ")
            elif(x == self.buildX and y == self.buildY):
                cprint(s, "white", "on_magenta", end =" ")
            else:
                print(s, end =" ")
          print()
env = Santorini()
#gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # self.conv1 = nn.Conv2d(5, 20, 3, padding=1)
        self.affine1 = nn.Linear(35, 20)
        self.dropout = nn.Dropout(p=0.2)
        self.affine2 = nn.Linear(80, 16)
        #self.dropout2 = nn.Dropout(p=0.6)
        # self.affine3 = nn.Linear(50, 50)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x = self.conv1(x)
        y1 = torch.cat((x[:,0,0], x[:,1,0], x[:,2,0], x[:,0,1], x[:,1,1], x[:,2,1], x[:,2,2]), 0)
#        x2 = torch.rot90(x, 1, [1, 2])
        y2 = torch.cat((x[:,4,0], x[:,4,1], x[:,4,2], x[:,3,0], x[:,3,1], x[:,3,2], x[:,2,2]), 0)
#        x3 = torch.rot90(x, 2, [1, 2])
        y3 = torch.cat((x[:,4,4], x[:,3,4], x[:,2,4], x[:,4,3], x[:,3,3], x[:,2,3], x[:,2,2]), 0)
#        x4 = torch.rot90(x, 3, [1, 2])
        y4 = torch.cat((x[:,0,4], x[:,0,3], x[:,0,2], x[:,1,4], x[:,1,3], x[:,1,2], x[:,2,2]), 0)
        x = torch.cat((self.affine1(y1), self.affine1(y2), self.affine1(y3), self.affine1(y4)), 0)
        # x =  torch.reshape(x, (1, 125))
        x = F.relu(x)
        # x = self.affine2(x)
        x = self.dropout(x)
        # x = F.relu(x)
        # x = self.affine2(x)
        # x = self.dropout2(x)
        # x = F.relu(x)        
        action_scores = torch.reshape(self.affine2(x), [2, 8])
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=3e-3)
#optimizer = optim.SGD(policy.parameters(), lr=0.000001, momentum=0.9)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action[0].item(), action[1].item()

#action[0].item()//5, action[0].item()%5, action[1].item()//5, action[1].item()%5


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def train(limit):    
    running_reward = 10
    running_steps = 1
    for i_episode in count(1):
        state, ep_reward, ep_steps = env.reset(), 0, 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                print("=========")
                print(action)
                env.render()
                print(reward, done)
            policy.rewards.append(reward)
            ep_reward += reward
            ep_steps = ep_steps + 1
            if done:
                break

        running_steps = 0.05 * ep_steps + (1 - 0.05) * running_steps
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tAverage steps: {:.2f}'.format(
                  i_episode, ep_reward, running_reward, running_steps))
        if limit > 0 and i_episode > limit:
            print("limit reached")
            break

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


def main():
    train(-1)
    policy.eval()
    env.seed(50)
    state, ep_reward, ep_steps = env.reset(), 0, 0
    for t in range(1, 10000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, _ = env.step(action)

        print("=========")
        print(action)
        env.render()
        print(reward, done)

        if done:
            break


def profmain():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        train(20)
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

if __name__ == '__main__':
    main()
