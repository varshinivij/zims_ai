from random import randint
from BoardClasses import Move
from BoardClasses import Board
import random, copy, math

class Node: 
    def __init__(self, board, move, parent, next_player):
        self.board = board 
        self.parent = parent # class is Node() represents the board before the most recent move was made
        self.children = []
        self.move = move # most recent move that led to this board state
        self.wins = 0
        self.visits = 0
        self.next_player = next_player # color of the player who is supposed to make a move now
        moves = board.get_all_possible_moves(next_player)
        self.moves = [item for sublist in moves for item in sublist] # flattened list of all possible moves for self.next_player from this board state

    def is_terminal(self):
        if self.parent and self.board.is_win(self.parent.next_player) != 0:
            return True
        return not self.moves
    
    def move_eval(self, board, move):
        score = 0
        #capture 
        if len(move.seq) > 2:
            score += 10
        #piece becomes king 
        if move.seq[-1][0] == 0 or move.seq[-1][0] == board.row - 1:
            score += 5
        return score
        
    def heuristic_move(self, curr_board, current_player):
        possible_moves = curr_board.get_all_possible_moves(current_player)
        possible_moves = [item for sublist in possible_moves for item in sublist]

        if not possible_moves: 
            return None
        
        board_evals = []
        total_score = 0

        for move in possible_moves:
            score = self.move_eval(curr_board, move)
            board_evals.append(score)
            total_score += score
            
        if total_score == 0:
            return random.choice(possible_moves)
        
        total_weight = sum(board_evals)
        r = random.uniform(0, total_weight)
        upto = 0
        selected_move = possible_moves[-1]

        for move, weight in zip(possible_moves, board_evals):
            if upto + weight >= r:
                selected_move = move
                break
            upto += weight

        return selected_move
    
    def capture_count(self, move, board, curr_player):
        #pass in and simulate curr_player move
        opponent = 1 if curr_player == 2 else 2
        sim_board = copy.deepcopy(board)
        sim_board.make_move(move, curr_player)

        #largest capture opponent has available 
        opp_moves = sim_board.get_all_possible_moves(opponent)
        opp_moves = [item for sublist in opp_moves for item in sublist]
        captures = 0
        
        for move in opp_moves:
            if len(move.seq) > 2:
                captures = max(captures,len(move.seq)-1)
        return captures #equivalent to max number of captures
        
    def get_move_by_capture_count(self, board, curr_player):
        idx = -1
        smallest_capture = float('inf')

        for i, move in enumerate(self.moves):
            ct = self.capture_count(move, board, curr_player)
            if ct < smallest_capture:
                smallest_capture = ct
                idx = i
        
        move = self.moves.pop(idx)
        return move

    def ucb(self): 
        if self.visits == 0: 
            return float("inf")
        return ((self.wins/self.visits) + (2**0.5) * ((math.log(self.parent.visits)/self.visits)**0.5))

    def select(self):
        return max(self.children, key=lambda child: child.ucb())
    
    def expansion(self): 
        new_move = self.get_move_by_capture_count(self.board, self.next_player) # choose one untried move -> convert to children
        new_board = copy.deepcopy(self.board)
        new_board.make_move(new_move, self.next_player)
        next_color = 1 if self.next_player == 2 else 2
        new_node = Node(new_board, new_move, self, next_color) # create new node with updated board state
        self.children.append(new_node) 
        return new_node

    def simulation(self): 
        curr_board = copy.deepcopy(self.board)
        current_player = self.next_player
        opponent = 1 if current_player == 2 else 2
        while True: # terminal condition: we keep on choosing moves until someone wins or both players are out of moves
            move = self.heuristic_move(curr_board, current_player)
            if not move: 
                break

            curr_board.make_move(move, current_player)
            if curr_board.is_win(current_player) != 0: 
                return curr_board.is_win(current_player)

            current_player = 2 if current_player == 1 else 1
            opponent = 2 if opponent == 1 else 1

        return -1

    def backpropogation(self, result): # result will store 1 or 2 for winner, -1 for tie
        curr_node = self 
        while curr_node:
            curr_node.visits += 1
            if result == -1: # its a tie
                curr_node.wins += 0.5
            elif curr_node.next_player != result: 
                curr_node.wins += 1
            curr_node = curr_node.parent 

class StudentAI():
    def __init__(self,col,row,p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col,row,p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2
    
    def get_move(self, move):
        if len(move) != 0:
            self.board.make_move(move,self.opponent[self.color])
        else:
            self.color = 1

        root = Node(copy.deepcopy(self.board), None, None, self.color)
        NUM_SIMULATIONS = 500

        for _ in range(NUM_SIMULATIONS):
            node = root
            # selection
            while not node.is_terminal() and not node.moves:
                node = node.select()
            # expansion
            if node.moves:
                node = node.expansion()

            # simulation
            result = node.simulation()
            node.backpropogation(result)
    
        best_child = max(
            root.children, 
            key=lambda child: child.wins / child.visits
        )

        if not best_child:
            return None

        self.board.make_move(best_child.move, self.color)
        move_to_return = best_child.move
        return move_to_return