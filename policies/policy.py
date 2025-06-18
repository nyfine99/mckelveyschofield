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
        self._values = np.array([float(value) for value in values])
        self.name = name

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        raise AttributeError("Cannot modify the value")
