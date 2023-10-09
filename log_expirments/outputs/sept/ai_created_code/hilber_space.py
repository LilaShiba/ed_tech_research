import numpy as np


class HilbertSpace:
    '''

    The `HilbertSpace` class has an 
    Please note that this is a basic implementation and may not cover all aspects of working with Hilbert spaces or frames.    
    '''

    def __init__(self, dim):
        '''
        `__init__` method that initializes the dimension of the Hilbert space 
        and an empty list to store the frames.

        '''
        self.dim = dim
        self.frames = []

    def add_frame(self, frame):
        '''
        The `add_frame` method allows you to add 
        frames to the Hilbert space. Each frame should 
        be a 1-dimensional NumPy array with the same 
        dimension as the Hilbert space.
        '''
        if len(frame) != self.dim:
            raise ValueError(
                "Frame dimension does not match Hilbert space dimension.")
        self.frames.append(frame)

    def inner_product(self, vec1, vec2):
        '''
        The `inner_product` method calculates the 
        inner product between two vectors in the Hilbert space. I
        t uses the dot product of each frame with the input vectors and sums them up.

        '''
        if len(vec1) != self.dim or len(vec2) != self.dim:
            raise ValueError(
                "Vector dimensions do not match Hilbert space dimension.")
        inner_prod = np.sum([frame.dot(vec1) * frame.dot(vec2)
                            for frame in self.frames])
        return inner_prod

    def norm(self, vec):
        '''
        The `norm` method calculates the norm of a 
        vector in the Hilbert space using the inner product.

        '''
        return np.sqrt(self.inner_product(vec, vec))

    def orthogonalize(self):
        '''
        The `orthogonalize` method orthogonalizes the frames using the QR decomposition.

        '''
        self.frames = np.linalg.qr(self.frames)[0]

    def normalize(self):
        '''
        The `normalize` method normalizes the frames by dividing each frame by its norm.

        '''
        self.frames = [frame / np.linalg.norm(frame) for frame in self.frames]
