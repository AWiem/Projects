from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2
    
    
    
    
    
#We create a function for the loop to find to split
def split_data(features, labels, t, feature_type):
    if feature_type == FeaturesTypes.REAL:
        s_features = [feature for feature in features if feature >= t]
        o_features = [feature for feature in features if feature < t]
    else:
        s_features = [feature for feature in features if feature == t]
        o_features = [feature for feature in features if feature != t]

    s_labels = [label for feature, label in zip(features, labels) if feature in s_features]
    o_labels = [label for feature, label in zip(features, labels) if feature in o_features]
   

    return s_features, o_features, s_labels, o_labels
    
    
    
    
class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.best_split = 0
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        """Computes the Gini score of the set of points
        Returns
        -------
        float
            The Gini score of the set of points
        """
        if len(self.labels) == 0: #we return zero if there are no labels
            return 0.0
        #We calculate Gini score for the labels
        data_length = len(self.labels)
        unique_l = []
        label_c = []
        for l in self.labels:
            if l in unique_l:
                label_c[unique_l.index(l)] += 1
            else:
                unique_l.append(l)
                label_c.append(1)
        gini = 1.0
        for i in label_c:
            p = i / data_length
            gini -= p ** 2

        return gini
        raise NotImplementedError('Please implement this function for Question 1')








    def get_best_gain(self, minimum=1) -> Tuple[int, float]:
        self.get_best_gain_called = True
        ginitab = []

        for i in range(len(self.features[0])):
            u_v = [self.features[j][i] for j in range(len(self.features))]
            u = list(set(u_v))

            max_gini_score = -1
            threshord = -1 
            s_features, o_features, s_labels, o_labels= [], [], [], []
            #We do the loop to find the ginigain for a split using that value as a threshold.
            for j in u:
                s_features, o_features, s_labels, o_labels = split_data(self.features[:, i], self.labels, j, self.types[i])
                        
                weight_s = len(s_features) / (len(s_features) + len(o_features))
                weight_o = len(o_features) / (len(s_features) + len(o_features))

                set_right = PointSet(s_features, s_labels, [])  # Create PointSet for the right subset
                set_left = PointSet(o_features, o_labels, [])   # Create PointSet for the left subset

                gain = weight_s * set_right.get_gini() + weight_o * set_left.get_gini()
                ginigain = self.get_gini() - gain
                #We determine max_gini_score and the threshold
                max_gini_score = max(max_gini_score, ginigain) if len(s_features) >= minimum and len(o_features) >= minimum else max_gini_score
                threshord = j if ginigain >= max_gini_score and len(s_features) >= minimum and len(o_features) >= minimum else threshord

            ginitab.append((max_gini_score, threshord))

        if not ginitab:
            return (None, None)

        maximal_gini, f_num, best_split = ginitab[0][0], 0, None

        for i, (gini, split) in enumerate(ginitab):
            if gini >= maximal_gini:
                maximal_gini, f_num, self.f_num, self.best_split = gini, i, i, split

        if self.best_split == -1:
            return (None, None)

        return (f_num, maximal_gini)

    def get_best_threshold(self) -> float:
        #We check if get_best_gain() was called, else we return None
        if not self.get_best_gain_called:
            return None
        #We check the feature type
        if self.types[self.f_num] == FeaturesTypes.REAL:
            feature_Vs = [data[self.f_num] for data in self.features]

            #Based on the best_split, we Separate feature values into left and right (LV: left values and RV: right values).
            LV = [value for value in feature_Vs if value < self.best_split]
            RV = [value for value in feature_Vs if value >= self.best_split]
            #Now we sort the values
            LV.sort()
            RV.sort()
            #We return the threshold value
            return (LV[-1] + RV[0]) / 2
        #Here we verify if the feature type is CLASSES and get_best_gain() was called, if it's the case then we return best_split
        elif self.types[self.f_num] == FeaturesTypes.CLASSES:
            if self.get_best_gain_called:
                return self.best_split
        
        return None






    
            
            


                