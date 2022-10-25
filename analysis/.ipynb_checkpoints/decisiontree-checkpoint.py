import numpy as np
import pandas as pd


from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from graphviz import Digraph, Source


thresholds = {'start_loop_upperhalf_col':[3,7,15,30,45],
              'highest_point_loop_upperhalf_col':[3,7,9,15,30,45],
              'gap_start':[2,8,20,40],
              'palindrome_score':[.25,.5,.6,.7,.8,.9],
              'large_asymmetric_bulge':[2,4,8,10,20,40,60],
              'largest_asym_bulge_sequence_location':[5,20,40,60,80],
              'stem_begin':[15,35,45,55,60,75],
              'stem_end':[3,5,8,12,20,30,40],
              'stem_length':[10,20,30,35,40,45,50,55,60,70,80,90],
              'total_length':[20,30,40,50,60,70,80,90],
              'base_pairs_in_stem':[.1,.3,.5,.7,.9],
              'base_pairs_wobbles_in_stem':[.1,.3,.5,.7,.9],
              'loop_width':[2,8,20,45],
              
              }


concept_list = ['presence_terminal_loop', 'start_loop_upperhalf_col',
       'highest_point_loop_upperhalf_row', 'highest_point_loop_upperhalf_col',
       'loop_length', 'loop_width', 'gap_start', 'palindrome_score',
       'asymmetric', 'large_asymmetric_bulge',
       'largest_asym_bulge_strand_location',
       'largest_asym_bulge_sequence_location', 'stem_begin', 'stem_end',
       'stem_length', 'total_length', 'base_pairs_in_stem',
       'base_pairs_wobbles_in_stem', 'AU_pair_begin_maturemiRNA', 'UGU']


class Node:
    
    def __init__(self, z, y, concepts, cntrl, parent=None):
        
        self.data = z
        self.y = y
        self.concepts = concepts
        self.rule = cntrl.nodenr
        self.nodenr = int(cntrl.nodenr)
        cntrl.nodenr += 1
        self.left_child = None
        self.right_child = None
        self.parent = parent
        self.leaf = False
        self.sibling = None

    def set_class(self, leaf):
        self.leaf = leaf
        self.mirna = np.round(np.mean(self.y))

    def delete_children(self):

        self.left_child.parent = None
        self.right_child.parent = None


        self.left_child = None
        self.right_child = None
        self.leaf = True

    def __eq__(self, other):
        return self.nodenr == other.nodenr

    
    @staticmethod
    def set_siblings(node1, node2):
        node1.sibling = node2
        node2.sibling = node1

class Controller:
    
    def __init__(self):
        
        self.nodenr = 0


class Tree:

    def __init__(self, thresholds, max_depth=5, min_gain=0.001,
                 min_samples=10, min_acc=0.8, finish_entropy=0.0001,
                 cls=LinearSVC, cls_args={'dual':False}):
        
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.min_samples = min_samples
        self.min_acc = min_acc
        self.finish_entropy = finish_entropy
        self.thresholds = thresholds
        self.cntrl = Controller()
        self.cls = cls
        self.cls_args = cls_args
        self.root = None

        self.depth_dict = {}
        for i in range(max_depth):
            self.depth_dict[i+1] = []
    

    def fit(self, X, y, concepts, prune=False):
        self.concept_list = list(concepts.columns)
        self.root = self.make_split(X, y, concepts, depth=0)
        if prune:
            self.prune()

    def create_thresholds(self, concepts):
        for i in self.concept_list:
            if np.unique(concepts[i]) > 11 and i not in self.thresholds:
                thresholds[i] = self.get_iqr(concepts[i])
    

    def get_iqr(self, col):
        qt = [col.quantile(0.25), col.quantile(0.5), col.quantile(0.75)]
        return list(np.unique(qt))


    def make_split(self, X, y, concepts, depth, parent=None):
        
        depth += 1
        entropy = self.calculate_entropy(y)
        node = Node(X, y, concepts, self.cntrl, parent)


        # if there are too little samples or if the entropy is low, stop splitting
        if X.shape[0] < self.min_samples or entropy < self.finish_entropy or depth == self.max_depth:
            node.set_class(True)
            return node

        best_rule = None
        best_cls = None
        e_gain = 0
    
        best = (best_cls, best_rule, e_gain)

        for i in self.concept_list:
            vals = np.unique(concepts[i]) 
            
            # if a concept is "pure"
            if len(vals) == 1:
                continue
            # if a concept is binary
            elif len(vals) == 2:
                rule = (concepts[i]>concepts[i].min()).values*1  
                best = self.calculate_split(X, rule, y, best, entropy, i)

            # if a concepts has a lot of values, use defined thresholds
            elif len(vals) > 11:
                for j in self.thresholds[i]:
                    rule = (concepts[i]>j).values*1
                    # if this results in all items having the same value
                    if len(np.unique(rule)) < 2:
                        continue
                    best = self.calculate_split(X, rule, y, best, entropy, i, j)
            # if a concept has a few values, try all of them
            # do not try the last one since nothing is bigger
            else:
                for j in vals[:-1]:
                    rule = (concepts[i]>j).values*1
                    best = self.calculate_split(X, rule, y, best, entropy, i, j)
            
        # if the new rule is not good enough, dont split
        best_cls, best_rule, e_gain = best
        if e_gain<self.min_gain:
            node.set_class(True)
            return node
        
            
        right = best_cls.predict(X)

        node.rule = best_rule
        node.rule_cls = best_cls

        # move on to children
        node.left_child = self.make_split(X[right==0], y[right==0],
                                          concepts[right==0], depth, node)
        node.right_child = self.make_split(X[right==1], y[right==1],
                                           concepts[right==1], depth, node)

        Node.set_siblings(node.right_child, node.left_child)

        self.depth_dict[depth].append((node.right_child, node.left_child))

        node.set_class(False)
        
        return node
        
    def calculate_split(self, X, concept, y, best, entropy, i, j=None):
        
        e_gain = best[2]

        kwargs = self.cls_args
        cls = self.cls(**kwargs)
        cls.fit(X, concept)
        pred = cls.predict(X)
        acc = np.mean(pred==concept)
        
        e_left = np.mean(pred==0) * self.calculate_entropy(y[pred==0]) if np.sum(pred==0) > 0 else 0
        e_right = np.mean(pred==1) * self.calculate_entropy(y[pred==1]) if np.sum(pred==1) > 0 else 0
        gain = entropy-(e_left+e_right)

        # if the rule is good, update splitting cirterion 
        if gain>e_gain and acc > self.min_acc:
            if j is None:
                rule = f'{i} is true'
            else:
                rule = f'{i} > {j}'
            return cls, rule, gain
        else:
            return best 
    
    def calculate_entropy(self, y, criterion='gini'):
        y1 = np.mean(y)
        y0 = 1 - y1
        if criterion == 'gini':
            entropy = 1 - (y0**2 + y1**2)
        elif criterion == 'entropy':
            entropy = -y0 * np.log2(y0) -y1 * np.log2(y1)
            
        return entropy


    def prune(self):
        for i in reversed(range(1, self.max_depth + 1)):
            for j in self.depth_dict[i]:
                self.prune_node(j)

    def prune_node(self, node_pair):
        node1 = node_pair[0]
        node2 = node_pair[1]

        if node1.leaf and node2.leaf and node1.mirna == node2.mirna:
            node1.parent.delete_children()

    def old_prune(self, node):
        '''
        removes redundant nodes
        '''

        # visit children
        if node.left_child is not None:
            self.prune(node.left_child)
        if node.right_child is not None:
            self.prune(node.right_child)

        sibling = node.parent.left_child if node == node.parent.right_child else node.parent.right_child
        if node.leaf and sibling.leaf and node.mirna == sibling.mirna:
            node.parent.delete_children()
        

    def predict(self, X):

        n = X.shape[0]
        y_hat = np.zeros((n,))
        for i in range(n):
            y_hat[i] = self.get_prediction(X[i].reshape(1,-1), self.root)

        return y_hat


    def get_prediction(self, X, node):
        if node.right_child is None:
            return node.mirna
        
        else:
            right = node.rule_cls.predict(X)
            if right == 1:
                return self.get_prediction(X, node.right_child)
            else:
                return self.get_prediction(X, node.left_child)


    def score(self, X, y):

        y_hat = self.predict(X)
        return np.mean(y_hat==y)


    def plot_tree(self):

        if self.root is None:
            print('No tree fitted!')

        else:
            g = Digraph()
            self.draw_node(self.root, level=0, g=g)
            return g

    def draw_node(self, node, level, g):        
        level+=1
        if node.left_child is not None and node.right_child is not None:
            g.node(f'{node.rule}{level}{node.nodenr}', f'{node.rule} \n{sum(node.y==0)}/{sum(node.y==1)}')
            self.draw_node(node.left_child, level, g)
            self.draw_node(node.right_child, level, g)
            g.edge(f'{node.rule}{level}{node.nodenr}', f'{node.left_child.rule}{level+1}{node.left_child.nodenr}', label='no')
            g.edge(f'{node.rule}{level}{node.nodenr}', f'{node.right_child.rule}{level+1}{node.right_child.nodenr}', label='yes')
            
        else:
            txt = 'miRNA' if sum(node.y==1) > sum(node.y==0) else 'non miRNA'
            g.node(f'{node.rule}{level}{node.nodenr}', f'{txt} \n{sum(node.y==0)}/{sum(node.y==1)}')
            