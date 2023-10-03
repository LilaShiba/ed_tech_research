import math
import numpy as np


class Vector:
    '''represent individual vectors in the Hilbert space.'''

    def __init__(self, probabilities):
        '''Each vector will have properties like its probability '''
        probs = np.array(probabilities)
        # convert vector to complex numbers
        self.probabilities = probs + 0j

    def normalize(self):
        '''L Norm'''
        norm = np.linalg.norm(self.probabilities)
        self.probabilities = self.probabilities / norm

    def collapse(self):
        ''' collapse wave function'''
        # Calculate the squared magnitudes of the complex components
        probabilities_squared = np.abs(self.probabilities)**2
        random_num = np.random.rand()
        cumulative_sum = np.cumsum(probabilities_squared)
        collapsed_index = np.argmax(cumulative_sum >= random_num)
        self.probabilities = np.zeros_like(
            self.probabilities, dtype=np.complex)
        self.probabilities[collapsed_index] = 1

    def cosine_similarity(self, other_vector):
        '''
        sub process for knn, compares cosines

        -1: exactly opposite in direction. 180-degree angle 

         0: perpendicular, with a 90-degree angle. 

         1: parallel, same direction.0-degree angle
        '''
        dot_product = np.dot(self.probabilities, other_vector.probabilities)
        norm_self = np.linalg.norm(self.probabilities)
        norm_other = np.linalg.norm(other_vector.probabilities)
        cosine_similarity = dot_product / (norm_self * norm_other)
        return cosine_similarity


if __name__ == "__main__":
    # Create some vectors for testing
    vecA = Vector([1, 2, 3])
    vecB = Vector([2, 3, 4])
    vecC = Vector([0, 0, 1])

    print("Original Vector A:", vecA.probabilities)
    print("Original Vector B:", vecB.probabilities)
    print("Original Vector C:", vecC.probabilities)
    print("-" * 40)

    # Test normalize()
    vecA.normalize()
    vecB.normalize()
    vecC.normalize()

    print("Normalized Vector A:", vecA.probabilities)
    print("Normalized Vector B:", vecB.probabilities)
    print("Normalized Vector C:", vecC.probabilities)
    print("-" * 40)

    # Test collapse()
    vecA.collapse()
    vecB.collapse()
    vecC.collapse()

    print("Collapsed Vector A:", vecA.probabilities)
    print("Collapsed Vector B:", vecB.probabilities)
    print("Collapsed Vector C:", vecC.probabilities)
    print("-" * 40)

# Test cosine_similarity()
sim_AB = vecA.cosine_similarity(vecB)
sim_BC = vecB.cosine_similarity(vecC)
sim_CA = vecC.cosine_similarity(vecA)

print("Cosine Similarity between A and B:", sim_AB)
print("Cosine Similarity between B and C:", sim_BC)
print("Cosine Similarity between C and A:", sim_CA)
