
import argparse
import numpy as np

a = """""
class Node:
    def __init__(self):
        self.d = None
        self.val = None
        self.left = None
        self.right = None

    def __init__(self, dimension, value, point):
        self.d = dimension
        self.val = value
        self.p = point
        self.left = None
        self.right = None

# Require a set of points P of M dimensions and current depth D
def BuildKdTree(P, D):
    # base case: Null and one data point
    if (P.val == None): # P is empty
        return None
    elif(P.size == 1): # P has one data point
        # create a new node
        # node.d = d
        # node.val = val
        # node.point = current.point
        return Node(D, P[0][D], P[0])
    else:
        # d <- D mod M
        d = D % M
        # val <- Median value along dimension d among points in P
        point = np.median(P, axis=d)
        val = point[d]
        # Create a new node
        # node.d <- d   node.val <- val
        # node.point <- point at the median along dimension d
        # node.left <- BuildKdTree(points in P for which value at dimension d is less than or equal to val, D+1)
        # node.right <- BuildKdTree(points in P for which value at dimension d is greater than val, D+1)
        ltoeq_points = []
        gt_points = []
        for points in P:
            if (points[d] >= val):
                ltoeq_points = 
            else:
                gt_points = P
        Node(d, val, point, BuildKdTree(ltoeq_points, D+1), BuildKdTree(gt_points, D+1))
        # return node
"""""

def main():

    parser = argparse.ArgumentParser(description="find the n nearest neighbour")

    #[train] specifies the path to a set of the training data file
    parser.add_argument('train', type=argparse.FileType('r'))

    #[test] specifies the path to a set of testing data file
    parser.add_argument('test', type=argparse.FileType('r'))

    #[dimension] is used to decide which dimension to start the comparison. (Algorithm 1)
    parser.add_argument('dimension', type=int)
    args = parser.parse_args()

    test_set = np.genfromtxt(args.test, names=True)
    train_set = np.genfromtxt(args.train, names=True)

if __name__ == "__main__":
    main()