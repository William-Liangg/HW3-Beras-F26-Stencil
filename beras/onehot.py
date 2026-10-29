import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_ones = np.unique(data)
        n = len(unique_ones) # number of classes
        
        eye_matrix = np.eye(n)
        
        self.encoding_dict = {
            label: eye_matrix[i] for i, label in enumerate(unique_ones)
        }

    def forward(self, data):
        list = [self.encoding_dict[label] for label in data]
        return np.array(list)

    def inverse(self, data):
        mapped = {np.argmax(vec): label for label, vec in self.encoding_dict.items()}
        max = np.argmax(data, axis=-1)
        
        labels = [mapped[idx] for idx in max]
        
        return np.array(labels)
