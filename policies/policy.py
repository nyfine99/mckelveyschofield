from typing import Union
import numpy as np

"""
A file which defines the Policy class. Once a policy is created, its values are immutable.
"""

class Policy():
    def __init__(self, values: Union[list[Union[float,int]], np.array], name: str = None):
        """
        Initializes a policy object.

        Params:
            values (list[float|int] | np.array): the values of the policy along each issue/axis.
        """
        self.name = name
        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError("Values must be a one-dimensional array.")
        elif isinstance(values, list):
            values = np.array(values, dtype=np.float64)
        else:
            raise TypeError("Values must be a list or a one-dimensional numpy array.")
        self._values = values
        self.id = self._values.tobytes()  # Unique identifier based on values

    @property
    def values(self):
        read_only_view = self._values.view()
        read_only_view.setflags(write=False)  # Make the view read-only
        return read_only_view

    @values.setter
    def values(self):
        raise AttributeError("Cannot modify the values of a Policy after it has been created.")
