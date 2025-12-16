import math
import random
EXPLORATION_CONST = math.sqrt(2)
MAX_SIMULATIONS = 100

# maintaining a set of visited states

def ucts(node):
    exploration = EXPLORATION_CONST * math.sqrt(math.log(node.parent.visits)/node.visits)
    exploitation = node.wins/node.visits
    return exploration + exploitation

def selection():
    node = board.root
    # start with the root node 
    while node in board.visited:
        max_ucts = float('-inf')
        for x, y in self.node.children:
            n = board.board[x][y]
            n_uct = ucts(n)
            if n_uct >= max_ucts:
                max_ucts = n_uct
                node = n

    return node


def expansion(n):
    visited.add(node.children)
    explore = random.randint(0, len(children))
    simulation(node.children[explore])


def simulation(node):
    while sims < MAX_SIMULATIONS and not node in self.terminal_states:
        move = random.randint(0, len(node.children))
        node = node.



def backpropogation():
    while node is not None:
        node.wins += 1
        node.visits += 1
        node = node.parent 
