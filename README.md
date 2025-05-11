# **Antichess Reinforcement Learning Agent**

A **Deep Q-Learning-based AI** for **Antichess**, a chess variant where the objective is to **lose all your pieces or get stalemated**. This project trains **two reinforcement learning agents** to compete against each other using delayed double Q-networks strategy.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Project Features](#project-features)
- [Rules of Antichess](#rules-of-antichess)
- [Reinforcement Learning Approach](#reinforcement-learning-approach)
- [Installation](#installation)
- [How to Train the Model](#how-to-train-the-model)
- [Testing the Trained Model](#testing-the-trained-model)
- [Results](#results)
- [References](#references)

---

## **Introduction**

Antichess, also known as **Losing Chess**, is a chess variant where players aim to **lose all their pieces** or **force a stalemate**. This project trains **a Deep Q-Network (DQN) agent** to play Antichess against another AI or a random strategy.

### **Key Components of the RL System:**
- **Deep Q-Learning** for decision-making.
- **Experience Replay** to stabilize training.
- **Polyak Averaging** to update target networks smoothly.
- **Self-play** for better learning.

---

## **Project Features**

✅ **Fully functional Antichess game logic**  
✅ **Deep Q-Learning with self-play training**  
✅ **Uses experience replay for stable learning**  
✅ **Implements Polyak averaging for smooth target network updates**  
✅ **Customizable opponent strategy (White or Black)**  

---

## **Rules of Antichess**

- The goal is to **lose all your pieces or get stalemated**.
- **Capturing is forced**—if a capture is available, the player must take it.
- **The king has no special status**—it can be captured like any other piece.
- **Pawns promote only to queens** upon reaching the last rank.

More details on Antichess rules: [Wikipedia](https://en.wikipedia.org/wiki/Losing_chess)

---

## **Reinforcement Learning Approach**

This project trains **two Deep Q-Learning agents** using **self-play**, enabling them to improve their strategy through thousands of games.

### **Key Techniques**

🔹 **Bellman Equation**:  
Used to update Q-values during training.  
🔹 **Deep Q-Networks (DQN)**:  
Neural network architecture with **experience replay** to prevent catastrophic forgetting.  
🔹 **Polyak Averaging**:  
Gradual target network updates for more stable learning.

### **Technical References**

📖 [Bellman Equation](https://www.cs.toronto.edu/~rgrosse/courses/csc311_f20/slides/lec11.pdf)  
📖 [Deep Q-Learning](https://blog.mlq.ai/deep-reinforcement-learning-q-learning/)  
📖 [Polyak Averaging](https://blog.mlq.ai/deep-reinforcement-learning-twin-delayed-ddpg-algorithm/)  

---

## **Installation**

### **Requirements**

- Python 3.8+
- PyTorch
- NumPy

### **Setup**

```bash
git clone https://github.com/pythagon-code/antichess-rl.git
cd antichess-rl
pip install -r requirements.txt
```

---

## **How to Train the Model**

Run the following command to train the AI:

```bash
python train.py
```

This will:
- Initialize the game board.
- Train two agents using **self-play**.
- Store experiences in **experience replay buffers**.
- Save trained models as `white.pth` and `black.pth`.

---

## **Testing the Trained Model**

To test the trained agent:

```bash
python test.py
```

- The trained AI plays against a **random strategy**.
- Set `agent_to_play = "white"` or `"black"` to control which side the AI plays.
- Results are displayed at the end.

---

## **Results**

- The trained AI achieved **76% accuracy as White** against a **random opponent**.
- The model demonstrated **strategic play** and improved convergence through **experience replay**.

---

## **References**

- 📖 **Bellman Equation**: [Understanding the Bellman Equation in Reinforcement Learning](https://www.cs.toronto.edu/~rgrosse/courses/csc311_f20/slides/lec11.pdf)  
- 📖 **Deep Q-Learning**: [Guide to Deep Q-Learning](https://blog.mlq.ai/deep-reinforcement-learning-q-learning/)  
- 📖 **Polyak Averaging**: [How Polyak Averaging Improves RL Stability](https://blog.mlq.ai/deep-reinforcement-learning-twin-delayed-ddpg-algorithm/)  

---

