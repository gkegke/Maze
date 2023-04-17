#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
import copy
import heapq
import math
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from colorama import Back
from enum import Enum, unique

from colors import Colors as C

@unique
class Things(Enum):
    EXIT = -2
    START = -1
    NOTHING = 0
    WALL = 1
    EXPLORING = 2
    DISCARDED = 3

class Maze:

    """

    A simple 2d-array representing the maze. 

    Each value of the 2d-array is a "Thing" (see Things class),
    such as wall or an unexplored room.

    DISPLAY_MAP - adds certain characters to a thing to a map
    THING_COLORS - colors the background of the thing a given color


    """

    DISPLAY_MAP = {
        Things.NOTHING : "{}   {}".format(Back.WHITE, Back.RESET),
        Things.WALL : "{} X {}".format(Back.RED, Back.RESET),
        Things.EXPLORING : "{} Y {}".format(Back.GREEN, Back.RESET),
        Things.DISCARDED : "{} Z {}".format(Back.YELLOW, Back.RESET),
        Things.START : "{} O {}".format(Back.WHITE, Back.RESET),
        Things.EXIT : "{} [] {}".format(Back.WHITE, Back.RESET)
    }

    THING_COLORS = {
        Things.NOTHING : (255, 255, 255), # white
        Things.WALL : (229, 20, 0), # red
        Things.EXPLORING : (0, 138, 0), # green
        Things.DISCARDED : (250, 104, 0), # orange
        Things.START : (0, 0, 0), # black
        Things.EXIT : (0, 0, 0) # black
    }

    def __init__(self, width=10, height=10, wall_pc=0.25):
        """

        Args:

         - width int , height int
           - dimensions of the 2d array maze

         - wall_pc
           - percent of values that'll be walls vs empty.

        """

        self.width = width
        self.height = height
        self.wall_pc = wall_pc
        self.maze = [[Things.NOTHING] * self.width for _ in range(self.height)]

        self.populate(wall_pc)

    def populate(self, wall_pc):
        """
        Add walls randomly to the maze based on a given percentage.
        """

        for i in range(self.width):
            for j in range(self.height):

                if random.random() < wall_pc:
                    self.maze[i][j] = Things.WALL

        self.maze[0][0] = Things.START
        self.maze[self.width-1][self.height-1] = Things.EXIT

    def repopulate(self):

        self.maze = [[Things.NOTHING] * self.width for _ in range(self.height)]

        self.populate(self.wall_pc)

    def display(self, _maze=None):
        """

        Helper function to generate the maze as a string to be displayed/printed
        for example testing in the console.

        """
        
        if _maze == None:
            _maze = self.maze

        output = []

        _maze_str = "\n".join([

            "".join([ self.DISPLAY_MAP[xy] for xy in row ])
            for row in _maze

        ])

        return _maze_str

    def runner(
        self,
        heuristics=("manhattan", "square", "euclidean"),
        allow_diagonal=True,
        fpath="result.gif",
        save_to_gif=True,
        _display_iterations=True
        ):
        """

        Helper function to attempt to solve the maze given a certain heuristics.

        Saves the progression as a gif.

        """

        def manhattan(current, target):
            return ( abs(target[0] - current[0]) + 
                     abs(target[1] - current[1]) ) 

        def square(current, target):
            return ((current[0] - target[0])**2 + 
                    (current[1] - target[1])**2) 

        def euclidean(current, target):
            return math.sqrt((current[0] - target[0])**2 + 
                    (current[1] - target[1])**2) 

        astar_heuristics = {
            "manhattan" : manhattan,
            "square" : square,
            "euclidean" : euclidean,
        }

        if len(heuristics) == 0:
            print("A heuristic needs to be chosen. Options are (manhattan, square, euclidean)")
            return

        results = dict()

        for h in heuristics:
            if h not in astar_heuristics:
                print(f"Heuristic {h} is not implemented")
            else:
                results[h] = dict()

        for heuristic in results:

            _found, _states = self.astar_solve(
                astar_heuristics[heuristic],
                allow_diagonal=allow_diagonal,
                _return_states=True,
                _display_iterations=_display_iterations
            )

            results[heuristic] = {
                "found" : _found,
                "_states" : _states
            }

        if save_to_gif:
            self.heuristics_to_gif(results, fpath)

        return results

    def simulator(
        self,
        runs=100
    ):
        """
        Run the heuristics N times, to get basic statistics of their performance
        relative to each other.

        Very simple for now.
        """
     
        results = dict()

        for _ in range(runs):
            r = self.runner(
                allow_diagonal=False,
                save_to_gif=False,
                _display_iterations=False
            )

            for h, hdata in r.items():
                
                if hdata["found"]:
                    if h not in results:
                        results[h] = [len(hdata["_states"])]
                    else:
                        results[h].append(len(hdata["_states"]))

            self.repopulate()

        s = "\n".join(
            [
                "{} : {}".format(
                    h,
                    sum(results[h])/len(results[h])
                )
                for h, n in results.items()
            ]
        )

        print(s)
        

    def heuristics_to_gif(self, data, fpath):
        """

        Saves results of all heuristics to a gif

        Useful for visual comparison

        TODO: Generalize a bit more, for example, the size/width of each block.

        """

        self.unicode_font = unicode_font = ImageFont.truetype("DejaVuSans.ttf", 30) 

        hnum = len(data)

        _temp_images = []

        # Generates the images for each heuristic
        for h, hdata in data.items():

            _images = []

            for a, maze_state in enumerate(hdata["_states"]):

                img = Image.new('RGB', (self.width*10, self.height*10), (255, 255, 255))
                d = ImageDraw.Draw(img)

                for i, maze_row in enumerate(maze_state):
                    for j, v in enumerate(maze_row):

                        print(v)

                        d.rectangle((i*10, j*10, i*10+10, j*10+10), fill=self.THING_COLORS[v])
                    
                d.text(
                  (100, 100),
                  h,
                  font=self.unicode_font,
                  fill=(0, 0, 0),
                )

                _images.append(img)

            _temp_images.append(_images)

        _max = max(
            len(image_set)
            for image_set in _temp_images
        )

        _final_images = []

        # combines them into a single vertical image
        for i in range(_max):

            merged_img = Image.new('RGB', (self.width*10, hnum * self.height * 10))

            for j, img_set in enumerate(_temp_images):

                if len(img_set) <= i:
                    merged_img.paste(img_set[-1], (0, j * self.height * 10))
                else:
                    merged_img.paste(img_set[i], (0, j * self.height * 10))

            _final_images.append(merged_img)

        # converts the images into a gif
        _final_images[0].save(fpath,
           save_all=True, append_images=_final_images[1:],
           optimize=False, duration=1000, loop=1)

    def astar_solve(
        self,
        heuristic_func,
        allow_diagonal=False,
        _return_states=True,
        _display_iterations=True
        ):
        """

        Args:

          heuristic_func function
           - A star requires a heuristic and different ones perform
             better is different situations.
           - Manhattan
           - Square
           - Euclidean

          allow_diagonal
           - Some heuristics will perform much better with this enabled.

          _return_states
           - return the states the maze was in during the running of the
           heuristic, so that it may be used/analyzed/visualized etc.

        A simple A* Search maze solver.

        There are optimizations that can be done such as bi-directional
        A* search (e.g. search from target to start at same time), but
        this is designed for fun hence they havn't been implemented (although
        maybe in the future).

        """

        maze = copy.deepcopy(self.maze)

        if allow_diagonal == False:

            _directions = (
                (-1, 0), # left
                (1, 0), # right
                (0, -1), # up
                (0, 1), # down
            )

        else:

            _directions = (
                (-1, 0), # left
                (1, 0), # right
                (0, -1), # up
                (0, 1), # down
                (-1, -1), # up left
                (1, -1), # down left
                (-1, 1), # up right
                (1, 1) # down right
            )

        # start location (0,0) => top, left
        curr = (0,0) # start

        # target location (n, m) here => bottom, right
        _target = (self.width - 1, self.height - 1)

        # hscore heuristic function
        H = heuristic_func

        # minheap to store discovered nodes
        # that need to be explored.
        # initially contains is the starting node
        explore_set = [(0, 0, curr)]

        # a set to track which node a node came from
        came_from = dict()

        # G[node] = total cost of cheapest path from start to node
        # default value infinity for easier processing
        G = defaultdict(lambda: float("inf")) 
        G[curr] = 0

        # f[node] = Gscore(node) + h(node) 
        # default value infinity for easier processing
        F = defaultdict(lambda: float("inf"))
        F[curr] = H(curr, _target)

        obstructed = set()

        if _return_states:
            states = [copy.deepcopy(maze)]
        else:
            states = None

        __found = False

        while explore_set != []:

            # get smallest 
            curr = heapq.heappop(explore_set)[2]
            maze[curr[0]][curr[1]] = Things.EXPLORING

            if curr == _target:
                __found = True
                break

            for dx, dy in _directions:

                cx, cy = curr
                neighbour = (cx + dx, cy + dy)

                if (neighbour in obstructed or
                    0 > neighbour[0] or
                    0 > neighbour[1] or
                    self.width <= neighbour[0] or
                    self.height <= neighbour[1] or
                    maze[neighbour[0]][neighbour[1]] == Things.WALL
                ):
                    obstructed.add(neighbour)
                    continue

                # Gscore of start to current node + distance of
                # neighbour to current node.
                # In this scenario, distance is just 1.
                # Can be different in more complex scenarios.
                temp_g = G[curr] + 1

                if temp_g < G[neighbour]:
                    came_from[neighbour] = curr
                    G[neighbour] = temp_g
                    F[neighbour] = temp_g + H(neighbour, _target)

                    if neighbour not in explore_set:
                        heapq.heappush(explore_set, (F[neighbour], G[neighbour], neighbour))

            if _return_states:
                states.append(copy.deepcopy(maze))

            if _display_iterations:
                print(self.display(maze))
                time.sleep(0.3)

        if __found:
            return True, states
        else:
            return False, states

if __name__ == "__main__":

    m = Maze(30, 30, 0.28)
    #m.runner(allow_diagonal=False)
    m.simulator(runs=10)
    #h = "manhattan"
    #m.astar_solve(h, allow_diagonal=False)
    #m.astar_solve(h, allow_diagonal=True)
    #m.random_solve("maze.gif")
