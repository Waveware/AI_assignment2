
import sys
import argparse
import numpy as np
import pandas as pd
from math import dist
from math import ceil
from statistics import median

# trace value for debugging
tr = 0

s = sys.stdout

class Node:
    def __init__(self):
        self.d = None
        self.val = None
        self.p = np.array([False, False, False, False])
        self.left = None
        self.right = None
    def is_empty(self):
        if (self.d == None):
            return False
        if (self.val == None):
            return False
        if (self.p.any() == False):
            return False
        if (self.left == None):
            return False
        if (self.right == None):
            return False
        return True
    def is_leaf(self):
        if (self.left == None and self.right == None):
            return True
        return False

# KdTree algorithm from the assignment description
def BuildKdTree(P, D) -> Node:
    M = 11
    if(tr == 1):
        print("D:", str(D))
    # base case: Null and one data point
    if (P.size <= 0): # P is empty
        if (tr == 1):
            print("P is empty")
        return None
    
    elif(P.size == 1): # P has one data point
        d = D%M

        # create a new node
        N = Node()
        # node.d = d
        N.d = d

        df_P = pd.DataFrame(P)
        df_P_values = df_P.to_numpy()

        # node.val = val
        N.val = df_P_values[0][d]
        # node.point = current.point
        N.p = df_P_values[0]

        if (tr == 1):
            print('----leaf node reached')
        return N
    
    else:
        # dimension
        d = D % M

        # val is the median value
        # point is the point at the median value at dimension d
        df = pd.DataFrame(P)
        df.sort_values(by=df.columns.values[d])
        df.reset_index()
        median_index = median(df.index.values)
        if (P.shape[0] == 2):
            point = df.values[0]
            val = (point[d] + df.values[1][d])/2
        else:
            point = df.values[int(ceil(median_index))]
            val = median(df[df.columns[d]])
        
        if(tr == 1):
            print("point:", str(point), "val:", str(val), "dim:", str(d))
        
        i = 0
        df_P_sorted = df.to_numpy()
        ltoeq_points = []
        gt_points = []
        for p in range(len(df_P_sorted)):
            if (tr == 1):
                print(df_P_sorted[p][d])
            if (df_P_sorted[p][d] <= val):
                ltoeq_points.append(i)
                i+=1
            else:
                gt_points.append(i)
                i+=1
        if(tr == 1):        
            print("--ltoeq:", str(ltoeq_points))
            print("--gt:", str(gt_points))

        # Create a new node
        N = Node()
        # node.d <- d | node.val <- val | node.point <- point at the median along dimension d
        N.d = d
        N.val = val
        N.p = point

        # node.left <- BuildKdTree(points in P for which value at dimension d is less than or equal to val, D+1)
        left = BuildKdTree(P[ltoeq_points], D+1)

        # node.right <- BuildKdTree(points in P for which value at dimension d is greater than val, D+1)        
        right = BuildKdTree(P[gt_points], D+1)
        
        N.left = left
        N.right = right

        return N

# unused
def QueryTree(tree, query_point, trace):
    # trace will assist with debugging.
    if(trace == 1):
        print("Query point:", str(query_point))
    nearest_node = QueryTreeSearch(tree, query_point, 10000, None, trace)
    return nearest_node

# unused
def QueryTreeSearch(this_node, query_point, nearest_distance, nearest_node, trace):
    if (trace == 1):
        print("-Current Node Point:", str(this_node.p))
        print("-Nearest Distance:", nearest_distance)
    if (this_node.p.any()): # does this node exist
        # calculate distance and comparing with nearest
        if (trace == 1):
            print("--Calculating Distance of Q with C")
            print("--  query_point            =", str(query_point))
            print("--  this_node.p[0:11]      =", str(this_node.p[0:11]))
        distance = dist(query_point, this_node.p[0:11])
        if(trace == 1):
            print("--  distance from this_node to query =", distance)
        if (distance < nearest_distance):
            if(trace == 1):
                print("---updated nearest node and distance")
            nearest_node = this_node
            nearest_distance = distance

        if (this_node.left == None and this_node.right == None):
            # must be a leaf
            if(trace == 1):
                print("------ leaf node reached. Return.")
                print("------ nearest distance =", nearest_distance)
            return this_node
        else: 
            # Traverse to deeper nodes
            d = this_node.d # dimension
            check_query_value = query_point[d] <= this_node.val
            if (check_query_value):
                if(trace == 1):
                    print("-----checking left subtree...")
                return QueryTreeSearch(this_node.left, query_point, nearest_distance, nearest_node, trace)           
            else:
                if (trace == 1):
                    print("-----checking right subtree...")
                return QueryTreeSearch(this_node.right, query_point, nearest_distance, nearest_node, trace)            
    else:
        return nearest_node

#   Query NN Search from the lecture slide
def QueryNNSearch(tree, query):
    # set the current node to the tree (root)
    node = tree
    node_stack = [tree]
    if (tr == 2):
        print("Starting QNNSearch for", str(query))
        print("With node:", str(tree))
    # keep going down until a leaf node is reached
    while (node != None):
        if (node.is_leaf() == True):
            break
        if (tr == 2):
            print('Going down to:',  str(node), "| d:", str(node.d), "| p:", str(node.p))
        if (query[node.d] < node.val):
            node = node.left
            node_stack.append(node)
        else:
            node = node.right
            node_stack.append(node)         
    
    # leaf node is now current best node
    while node is None:
        node = node_stack.pop()

    if (tr == 2):
        print("best_node:", str(node), "| d:", str(node.d), "| p:", str(node.p))
    best_node = node
    best_distance = dist(query, best_node.p[0:11])
    if (tr == 2):
        print("leaf node reached! P:", str(node.p))
        print("leaf node best_distance:", str(best_distance))
    
    while (node != tree):
        if (len(node_stack) == 0):
            return best_node
        node = node_stack.pop() # current node
        if (node.is_empty()):
            continue

        lowerbound_distance = query[node.d] - node.val
        if (lowerbound_distance < 0):
            check_distance = dist(query, node.p[0:11])
            if (best_distance > check_distance):
                best_distance = check_distance
                best_node = node
            if (query[node.d] < node.val):
                if(node.left == None):
                    continue
                else:
                    if (tr == 2):
                        print("going left")
                    node = node.left
                    node_stack.append(node)
                if (node.right == None):
                    continue
                else:
                    if(tr == 2):
                        print("going right")
                    node = node.right
                    node_stack.append(node)         

    
    if(tr == 2):
        print("root node is reached!")

    return best_node



def main():

    parser = argparse.ArgumentParser(description="find the n nearest neighbour")

    #[train] specifies the path to a set of the training data file
    parser.add_argument('train', type=argparse.FileType('r'))

    #[test] specifies the path to a set of testing data file
    parser.add_argument('test', type=argparse.FileType('r'))

    #[dimension] is used to decide which dimension to start the comparison. (Algorithm 1)
    parser.add_argument('dimension', type=int)
    args = parser.parse_args()

    train_set = np.genfromtxt(args.train, names=True)
    test_set = np.genfromtxt(args.test, names=True)

    tsdf = pd.DataFrame(test_set)
    ts = tsdf.to_numpy()

    dim = args.dimension

    # construct tree
    tree = BuildKdTree(train_set, dim)

    # input test on trained KdTree and find 1NN for each sample.
    wine_quality_predictions = []
    for wine in range(ts.shape[0]):
        #closest_wine_approximation = QueryTree(tree, ts[wine], tr)
        closest_wine_approximation = QueryNNSearch(tree, ts[wine])
        wine_quality_predictions.append(closest_wine_approximation.p[-1])

    #print("Wine predictions:")
    for wine in wine_quality_predictions:
        print(int(wine))


if __name__ == "__main__":
    main()