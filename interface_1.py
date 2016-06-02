# -*- coding: utf-8 -*-
"""
Test script to see if the import from .solver import graph_fused_lasso_warm works

Please declare the variables, args.*, y, edges, ntrails, trails, breakpoints
even in an external csv file.
"""
import csv
import numpy as np
from collections import defaultdict
from gfl.solver import pygfl, pygfl_weight, pygfl_augmented

def load_trails(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        trails = []
        breakpoints = []
        edges = defaultdict(list)
        for line in reader:
            if len(trails) > 0:
                breakpoints.append(len(trails))
            nodes = [int(x) for x in line]
            trails.extend(nodes)
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
        if len(trails) > 0:
            breakpoints.append(len(trails))
    return (len(breakpoints), np.array(trails, dtype="int32"), np.array(breakpoints, dtype="int32"), edges)

class TrailSolver:
    def __init__(self, alpha=2., inflate=2., maxsteps=1000000, converge=1e-6):
        self.alpha = alpha
        self.inflate = inflate
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, y, edges, ntrails, trails, breakpoints, weights=None):
        self.y = y
        self.edges = edges
        self.nnodes = len(y)
        self.ntrails = ntrails
        self.trails = trails
        self.breakpoints = breakpoints
        self.weights = weights
        self.beta = np.zeros(self.nnodes, dtype='double')
        self.z = np.zeros(self.breakpoints[-1], dtype='double')
        self.u = np.zeros(self.breakpoints[-1], dtype='double')
        self.steps = []

    def solve(self, lam, lam2=0):
        '''Solves the GFL for a fixed value of lambda.'''
        if self.weights == None:
            s = pygfl(self.nnodes, self.y,
                        self.ntrails, self.trails, self.breakpoints,
                        lam,
                        self.alpha, self.inflate, self.maxsteps, self.converge,
                        self.beta)
        elif lam2 == 0:
            s = pygfl_weight(self.nnodes, self.y, self.weights,
                                self.ntrails, self.trails, self.breakpoints,
                                lam,
                                self.alpha, self.inflate, self.maxsteps, self.converge,
                                self.beta)
        else:
            s = pygfl_augmented(self.nnodes, self.y, self.weights,
                                self.ntrails, self.trails, self.breakpoints,
                                lam, lam2,
                                self.alpha, self.inflate, self.maxsteps, self.converge,
                                self.beta)                        
        return self.beta
       
        
if __name__ == "__main__":
    # TODO: initiate arguments of TrailSolver
    # this is the call from the __init__ methjod in gfl
    alpha = 2.
    inflate = 2.
    maxsteps=1000000
    converge=1e-6
    lam = 1.
    y = np.loadtxt('residuals.csv', delimiter=',')
    weights = np.loadtxt('weights.csv',delimiter=',')
    ntrails, trails, breakpoints, edges = load_trails('trails.csv')
    solver = TrailSolver(alpha, inflate, maxsteps, converge)
    # Set the data and pre-cache any necessary structures
    solver.set_data(y, edges, ntrails, trails, breakpoints, weights)

    beta = solver.solve(lam, lam2=.1)
