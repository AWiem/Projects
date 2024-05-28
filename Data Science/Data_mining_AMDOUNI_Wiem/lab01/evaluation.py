from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    TP = sum(1 for e, a in zip(expected_results, actual_results) if e and a)
    PP = sum(actual_results)
    AP = sum(expected_results)

    precision = TP / PP if PP > 0 else 0.0
    recall = TP / AP if AP > 0 else 0.0

    return precision, recall
    #raise NotImplementedError('Implement this method for Question 3')

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)

    if (precision + recall) == 0:
        F1 = 0.0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)

    return F1
    #raise NotImplementedError('Implement this method for Question 3')

