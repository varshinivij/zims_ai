from BoardClass import is_valid_coordinate, compute_coordinates 
class Node:
    def __init__(self, x, y, n):
        self.wins = 0
        self.visits = 0
        self.parent = None
        self.children = []
        self.untried_moves = []
        self.position = (x, y)
        self.n = int(n) #board dimensions

    def process_children():
        coordinates = compute_coordinates(self.position)
        for coords in coordinates:
            if is_valid_coordinate(coords) and coords != self.parent:
                self.untried_moves.append(coords)
        