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

def is_expanded():
    if len(node.untried_moves) == 0:
        return True
    return False

def expansion(n):
    move = random.choice(n.untried_moves)
    n.untried_moves.remove(move)
    n.children.append(move)
    simulation(move)


def simulation(start_node):
    node = start_node
    while sims < MAX_SIMULATIONS and not node in self.terminal_states:
        move = random.randint(0, len(node.children))
        node = node.children[move]
        board.make_move(node)
    
    backpropogation(start_node)


def backpropogation(node):
    while node is not None:
        node.wins += 1
        node.visits += 1
        node = node.parent 
