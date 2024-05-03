
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

# Require a set of points P of M dimensions and current depth D
def BuildKdTree(P, D) -> Node:
    M = 11
    if(tr == 1):
        print("D:", str(D))
    # base case: Null and one data point
    if (P.size <= 0): # P is empty
        if (tr == 1):
            print("P is empty")
        return Node()
    
    elif(P.size == 1): # P has one data point
        # create a new node
        # node.d = d
        # node.val = val
        # node.point = current.point
        d = D%M

        N = Node()
        N.d = d

        df_P = pd.DataFrame(P)
        df_P_values = df_P.to_numpy()

        N.p = df_P_values[0]
        N.val = df_P_values[0][d]
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
            val = point[d]
        
        if(tr == 1):
            print("point:", str(point), "val:", str(val), "dim:", str(d))
        
        # Create a new node
        # node.d <- d   node.val <- val
        # node.point <- point at the median along dimension d
        # node.left <- BuildKdTree(points in P for which value at dimension d is less than or equal to val, D+1)
        # node.right <- BuildKdTree(points in P for which value at dimension d is greater than val, D+1)
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
        N = Node()
        N.d = d
        N.val = val
        N.p = point
        left = BuildKdTree(P[ltoeq_points], D+1)
        right = BuildKdTree(P[gt_points], D+1)
        N.left = left
        N.right = right
        #print(N)
        return N

# Require the root of the KdTree, point of query, and trace value (0 or 1)
def QueryTree(tree, query_point, trace):
    # trace will assist with debugging.
    if(trace == 1):
        print("Query point:", str(query_point))
    nearest_node = QueryTreeSearch(tree, query_point, 10000, None, trace)
    return nearest_node

# Recursive function called by Query Tree
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
        closest_wine_approximation = QueryTree(tree, ts[wine], tr)
        wine_quality_predictions.append(closest_wine_approximation.p[-1])

    #print("Wine predictions:")
    i = len(wine_quality_predictions)-1
    while(i > -1):
        print(int(wine_quality_predictions[i]))
        i -= 1


if __name__ == "__main__":
    main()