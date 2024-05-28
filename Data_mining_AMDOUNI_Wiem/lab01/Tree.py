from typing import List
from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        #We initialize our parametres

        self.features = features
        self.labels = labels
        self.types = types
        self.h = h
        self.left_tree = None
        self.right_tree = None
        self.new_threshold = None
        self.min_split_points = min_split_points

        #We initialize 4 lists that will contain information about left_tree and right_tree

        labels_l = []
        labels_r = []
        features_r = []
        features_l = []

        # Calculate the best feature to use to split
       
        self.points = PointSet(features, labels, types)
        self.feature = self.points.get_best_gain(min_split_points)[0]
        
        # Verify that there is no unique value in the labels
       
        verify_uniquness = (len(list(set(labels))) == 1)
      
        #Now, based on some conditions,  we split
        
        if self.feature != None and h >0 and not verify_uniquness  :
           
            # If the type is not BOOLEAN then, we can access to the best threshold
            
            if self.types[self.feature] != FeaturesTypes.BOOLEAN:
                self.new_threshold = self.points.get_best_threshold()
            
            # Else the best split is the only split possible after all
           
            else:
                self.new_threshold = self.points.best_split
           
            
            for i in range(len(features)):
            
             # We first define some conditions to use for the splitting
                
                if self.types[self.feature] == FeaturesTypes.REAL:
                    condition = (features[i][self.feature] >=  self.new_threshold)
                else:
                    condition = (features[i][self.feature] == self.new_threshold)
                
                
                if condition:
                    labels_r.append(labels[i])
                    features_r.append(features[i])
                else:
                    labels_l.append(labels[i])
                    features_l.append(features[i])
            # Construct the right and left tree
            self.left_tree = Tree(features_l,labels_l,types,h-1,min_split_points)
            self.right_tree = Tree(features_r,labels_r,types,h-1,min_split_points)

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        #If we reach a leaf, we have left and right trees null,
        #we predict based on the number of true and false that we got

        if self.left_tree is None:
            count_true = 0
            count_false = 0
            for i in self.labels:
                if i == True:
                    count_true=count_true+1 
                else:
                    count_false=count_false+1
            return (count_true >= count_false)
        
        # If we don't reach a leaf, then we split based on the comparaison of our feature to the threshold
        # We also define some conditions like previously
       
        else:
            if self.types[self.feature] == FeaturesTypes.REAL:
                condition = (features[self.feature] >= self.new_threshold)
            else:
                condition = (features[self.feature] == self.new_threshold)


            
            if condition:
                return self.right_tree.decide(features)
            else:
                return self.left_tree.decide(features)

