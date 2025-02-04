####################################################################################
#                              Antichess Logic                                     #
####################################################################################

initial_pos = [["r", "n", "b", "q", "k", "b", "n", "r"]] + [["p"] * 8] + [["."] * 8] * 4 + [["P"] * 8] + [["R", "N", "B", "Q", "K", "B", "N", "R"]]

import numpy as np
from random import choice, sample

def printBoard(board):
  piece_to_emoji = {
      "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚", "p": "♟",
      "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "P": "♙",
      ".": "▇"
  }

  for row in board:
      print(" ".join(piece_to_emoji[piece] for piece in row))

def isWhite(ch):
  if ch == ".":
    return False
  return ch < "a"

def isValidSpace(row, col):
  return 0 <= row < 8 and 0 <= col < 8

def isEmpty(ch):
  return ch == "."

def isValidSpaceAndIsEmpty(board, row, col):
  return isValidSpace(row, col) and isEmpty(board[row][col])

def isOppositeColor(ch1, ch2):
  return ch1 != "." and ch2 != "." and isWhite(ch1) != isWhite(ch2)

def getPawnMoves(board, row, col):
  if isWhite(board[row][col]):
    add = -1
  else:
    add = 1
  moves = []
  if isValidSpaceAndIsEmpty(board, row + add, col):
    moves.append((row, col, row + add, col))
    if row == 6 if isWhite(board[row][col]) else row == 1:
      if isValidSpaceAndIsEmpty(board, row + 2 * add, col):
        moves.append((row, col, row + 2 * add, col))
  if isValidSpace(row + add, col + 1) and isOppositeColor(board[row][col], board[row + add][col + 1]):
    moves.append((row, col, row + add, col + 1))
  if isValidSpace(row + add, col - 1) and isOppositeColor(board[row][col], board[row + add][col - 1]):
    moves.append((row, col, row + add, col - 1))
  return moves

def RookOrBishopMoves(board, row, col, rowincr, colincr):
  newrow = row + rowincr
  newcol = col + colincr
  moves = []
  while isValidSpace(newrow, newcol):
    if isOppositeColor(board[row][col], board[newrow][newcol]):
      moves.append((row, col, newrow, newcol))
    if not isEmpty(board[newrow][newcol]):
      break
    moves.append((row, col, newrow, newcol))
    newrow += rowincr
    newcol += colincr
  return moves

def getRookMoves(board, row, col):
  moves = RookOrBishopMoves(board, row, col, -1,  0)
  moves += RookOrBishopMoves(board, row, col,  1,  0)
  moves += RookOrBishopMoves(board, row, col,  0, -1)
  moves += RookOrBishopMoves(board, row, col,  0,  1)
  return moves

def getBishopMoves(board, row, col):
  moves = RookOrBishopMoves(board, row, col, -1, -1)
  moves += RookOrBishopMoves(board, row, col, -1,  1)
  moves += RookOrBishopMoves(board, row, col,  1, -1)
  moves += RookOrBishopMoves(board, row, col,  1,  1)
  return moves

def getQueenMoves(board, row, col):
  return getRookMoves(board, row, col) + getBishopMoves(board, row, col)

def getKnightMoves(board, row, col):
  rowincr = [-1, -1, +1, +1, -2, -2, +2, +2]
  colincr = [-2, +2, -2, +2, -1, +1, -1, +1]
  moves = []
  for i in range(len(rowincr)):
    newrow = row + rowincr[i]
    newcol = col + colincr[i]
    if isValidSpace(newrow, newcol):
      if isEmpty(board[newrow][newcol]) or isOppositeColor(board[row][col], board[newrow][newcol]):
        moves.append((row, col, newrow, newcol))
  return moves

def getKingMoves(board, row, col):
  rowincr = [-1, -1, -1,  0,  0, +1, +1, +1]
  colincr = [-1,  0, +1, -1, +1, -1,  0, -1]
  moves = []
  for i in range(len(rowincr)):
    newrow = row + rowincr[i]
    newcol = col + colincr[i]
    if isValidSpace(newrow, newcol):
      if isEmpty(board[newrow][newcol]) or isOppositeColor(board[row][col], board[newrow][newcol]):
        moves.append((row, col, newrow, newcol))
  return moves

def getAllMoves(board, white):
  moves = []
  for r in range(len(board)):
    for c in range(len(board[0])):
      piece = board[r][c]
      if not isEmpty(piece) and isWhite(piece) == white:
        if piece.lower() == "p":
          moves += getPawnMoves(board, r, c)
        elif piece.lower() == "r":
          moves += getRookMoves(board, r, c)
        elif piece.lower() == "n":
          moves += getKnightMoves(board, r, c)
        elif piece.lower() == "b":
          moves += getBishopMoves(board, r, c)
        elif piece.lower() == "q":
          moves += getQueenMoves(board, r, c)
        elif piece.lower() == "k":
          moves += getKingMoves(board, r, c)

  capture = list(filter(lambda move: isOppositeColor(board[move[0]][move[1]], board[move[2]][move[3]]), moves))
  if len(capture) > 0:
    moves = capture

  return moves

def playMove(board, move):
  r0, c0, r1, c1 = move
  if board[r0][c0] == "p" and r1 == 7: # Black pawn promotion
    board[r0][c0] = "q"
  elif board[r0][c0] == "P" and r1 == 0: # White pawn promotion
    board[r0][c0] = "Q"
  board[r1][c1] = board[r0][c0]
  board[r0][c0] = "."

board = np.array(initial_pos)

printBoard(board)

####################################################################################
#                              Deep Q-Learning                                     #
####################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

n_neurons = 64
lr = 0.01
tau = 0.05
epsilon_decay = 0.99
min_epsilon = 0.1
n_games = 1000
gamma = 0.995
buffer_size = 5000
batch_size = 64

class QNet(nn.Module):
  def __init__(self, n=n_neurons):
    super(QNet, self).__init__()
    self.fc1 = nn.Linear(128, n*4)
    self.fc2 = nn.Linear(n*4, n*2)
    self.fc3 = nn.Linear(n*2, n)
    self.fc4 = nn.Linear(n, 1)
    self.elu = nn.ELU()

  def forward(self, x):
    x = self.elu(self.fc1(x))
    x = self.elu(self.fc2(x))
    x = self.elu(self.fc3(x))
    x = self.fc4(x)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

policy_net_white = QNet().to(device)
target_net_white = QNet().to(device)
target_net_white.load_state_dict(policy_net_white.state_dict())

policy_net_black = QNet().to(device)
target_net_black = QNet().to(device)
target_net_black.load_state_dict(policy_net_black.state_dict())

white_optimizer = optim.Adam(policy_net_white.parameters(), lr=lr)
black_optimizer = optim.Adam(policy_net_black.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Polyak averaging
def update_target_net(policy, target, tau=tau):
  for target_param, policy_param in zip(target.parameters(), policy.parameters()):
      target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

def getBestMove(board, moves, net):
  board2 = board.copy()
  state_actions = np.empty((len(moves), 128))

  for i, move in enumerate(moves):
    playMove(board2, move)

    old_board_state = [ord(ch) for ch in board.flatten()]
    new_board_state = [ord(ch) for ch in board2.flatten()]

    state_actions[i] = old_board_state + new_board_state

  state_actions = torch.tensor(state_actions, dtype=torch.float32, device=device)
  with torch.no_grad():
    rewards = net(state_actions)
  return moves[torch.argmax(rewards)]

def store_experience(dq, state_action, next_state_action, reward):
    dq.append((state_action, next_state_action, reward))

def sample_batch(dq, batch_size=batch_size):
  return sample(dq, batch_size)

def initialize_board():
  return np.array(initial_pos)

# Antichess rules: Lose all your pieces or get stalemated
def calculate_reward_and_predict_next_state(board, moves, net, turn):
  board = board.copy()
  opponent_moves = getAllMoves(board, turn)
  if len(opponent_moves) == 0:
    # If None is the next state, we expect to skip the training phase as we cannot apply Bellman's equation
    return -1, None # Opponent wins (They lost all their pieces or were stalemated)

  opponent_move = getBestMove(board, opponent_moves, net)

  playMove(board, opponent_move)

  agent_moves_after = getAllMoves(board, not turn)
  if len(agent_moves_after) == 0:
    # In this case, we cannot apply Bellman's equation because we cannot play any moves after this terminal state
    return 1, None # Agent wins (You lost all your pieces or are stalemated)

  return 0, board

dq_white = deque(maxlen=buffer_size)
dq_black = deque(maxlen=buffer_size)

predicted_q_values = torch.tensor([np.nan], device=device); # For debugging
epsilon = 0.5

# Training loop
for episode in range(n_games):
  board = initialize_board()
  turn = True
  done = False

  i = 0
  won = None
  while not done and i < 800:
    i += 1
    moves = getAllMoves(board, turn)
    if len(moves) == 0:
      done = True
      won = "White" if turn else "Black"
      epsilon = max(min_epsilon, epsilon * epsilon_decay)
      break

    if np.random.random() < epsilon:
      agent_move = choice(moves)
    else:
      agent_move = getBestMove(board, moves, policy_net_white if turn else policy_net_black)
    board2 = board.copy()
    playMove(board2, agent_move)
    state_action = np.concatenate([np.array([ord(ch) for ch in board.flatten()]), [ord(ch) for ch in board2.flatten()]])
    board = board2

    reward, predicted_next_state = calculate_reward_and_predict_next_state(board, moves, policy_net_white if not turn else policy_net_black, not turn)
    if predicted_next_state is None:
      next_state_action = np.zeros_like(state_action)
    else:
      next_moves = getAllMoves(predicted_next_state, turn)

      next_move = getBestMove(predicted_next_state, next_moves, policy_net_white if not turn else policy_net_black)
      next_next_state = predicted_next_state.copy()
      playMove(next_next_state, next_move)

      old_board_state = [ord(ch) for ch in predicted_next_state.flatten()]
      new_board_state = [ord(ch) for ch in next_next_state.flatten()]

      next_state_action = np.concatenate([old_board_state, new_board_state])

    store_experience(dq_white if not turn else dq_black, state_action, next_state_action, reward)

    # Polyak averaging
    update_target_net(policy_net_white if turn else policy_net_black, target_net_white if turn else target_net_black)

    # Replay experience
    if len(dq_white) > batch_size if turn else len(dq_black) > batch_size:
      batch = sample_batch(dq_white if turn else dq_black)

      state_actions, next_state_actions, rewards = zip(*batch)
      state_actions = torch.tensor(state_actions, dtype=torch.float32, device=device)
      next_state_actions = torch.tensor(next_state_actions, dtype=torch.float32, device=device)
      rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

      predicted_q_values = (policy_net_white if turn else policy_net_black)(state_actions)
      with torch.no_grad():
        future_q_values = (target_net_white if turn else target_net_black)(next_state_actions)
      target_q_values = rewards.unsqueeze(1) + gamma * future_q_values  # Bellman equation
      loss = loss_fn(predicted_q_values, target_q_values)

      optimizer = white_optimizer if turn else black_optimizer
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    turn = not turn

  print(f"Episode {episode + 1}/{n_games} - Reward: {predicted_q_values.mean()},  Won by: {won}")
  printBoard(board)

def save_models(model, filename):
    torch.save(model.state_dict(), filename)

def load_models(model, filename):
    model.load_state_dict(torch.load(filename))

save_models(policy_net_white, "white.pth")
save_models(policy_net_black, "black.pth")

printBoard(board)

####################################################################################
#                              Testing        c:\Users\abhay_7d5ltou\Downloads\antichess_reinforcement_learnig.py                                     #
####################################################################################
from time import sleep

agent_won = 0
total_games = 100

for x in range(total_games):
  board = initialize_board()
  printBoard(board)

  agent_to_play = "black"  # Set to "white" or "black"
  trained_net = policy_net_white if agent_to_play == "white" else policy_net_black  # Assign correct network

  white = True
  i = 0

  while True and i < 400:
      i += 1
      moves = getAllMoves(board, white)

      if len(moves) == 0:
          print(f"{'White' if white else 'Black'} won")
          if white:
            agent_won += 1
          break

      if white == (agent_to_play == "white"):  # If it's the agent's turn
          move = getBestMove(board, moves, trained_net)
      else:  # Opponent plays randomly
          move = choice(moves)

      playMove(board, move)
      printBoard(board)
      print()

      white = not white  # Switch turn

print(f"Agent won {agent_won} out of {total_games} games.")