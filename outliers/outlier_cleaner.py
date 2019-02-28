#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    n = len(predictions)
    temp = []
    for i in range(n):
        temp.append((ages[i], net_worths[i], abs(predictions[i] - net_worths[i]) ))
    
    temp = sorted(temp, key = lambda x: x[2] )
    cleaned_data = temp[:int(n*0.9)]

    return cleaned_data

