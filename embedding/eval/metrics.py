import numpy as np
from scipy.spatial.distance import cosine

def cosine_similarity(x, y):
    # # Ensure length of x and y are the same
    # if len(x) != len(y) :
    #     return None
    
    # # Compute the dot product between x and y
    # dot_product = np.dot(x, y)
    
    # # Compute the L2 norms (magnitudes) of x and y
    # magnitude_x = np.sqrt(np.sum(x**2)) 
    # magnitude_y = np.sqrt(np.sum(y**2))
    
    # # Compute the cosine similarity
    # cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    # print(cosine_similarity)

    cosine_similarity = []
    for i in range(x.shape[0]):
        s = 1 - cosine(x[i], y[i])
        cosine_similarity.append(s)

    return np.array(cosine_similarity)