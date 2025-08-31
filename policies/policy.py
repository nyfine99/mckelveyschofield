"""
Policy Data Structure Module

This module defines the Policy class, which represents policy positions in multi-dimensional
policy spaces. Policies are fundamental data structures used throughout the electoral
dynamics system to represent candidate positions, voter preferences, and policy proposals.

Key Features:
- Immutable policy values after creation (enforces data integrity)
- Multi-dimensional policy space support (typically 2D for electoral simulations)
- Unique identification based on policy values
- Read-only access to policy coordinates
- Support for both list and numpy array inputs

The Policy class is designed to be lightweight yet robust, providing a reliable
foundation for electoral calculations, winset analysis, and McKelvey-Schofield
pathfinding algorithms.
"""

from typing import Union
import numpy as np


class Policy():
    """
    Immutable policy object representing a position in multi-dimensional policy space.
    
    The Policy class encapsulates policy positions as coordinate vectors, where each
    coordinate represents a position on a specific policy dimension (e.g., economic
    policy, social policy, foreign policy). Once created, policy values cannot
    be modified, ensuring data integrity throughout electoral simulations.
    
    Policies are used throughout the system to represent:
    - Candidate/party policy positions
    - Voter ideal points
    - Policy proposals
    - Status quo positions for comparison
    
    The class automatically handles input validation and type conversion, and provides
    a unique identifier based on the policy values for efficient comparison and
    storage operations.
    
    Attributes:
        name (str, optional): Human-readable identifier for the policy.
                             Useful for labeling plots and debugging.
        _values (np.ndarray): Private array storing the actual policy coordinates.
                              Always one-dimensional with float64 dtype.
        id (bytes): Unique identifier derived from policy values.
                    Used for efficient equality comparisons and hashing.
    """

    def __init__(self, values: Union[list[Union[float,int]], np.array], name: str = None):
        """
        Initialize a policy object with coordinate values.
        
        Creates a new Policy object representing a position in policy space.
        The input values are automatically converted to a one-dimensional numpy
        array and validated for proper dimensionality. A unique identifier is
        generated based on the coordinate values for efficient comparison.
        
        Args:
            values (list[float|int] | np.array): Policy coordinates along each dimension.
                                                Must be one-dimensional and contain
                                                numeric values. Typical electoral
                                                simulations use 2D policy spaces.
            name (str, optional): Human-readable identifier for the policy.
                                 Useful for debugging, plotting, and documentation.
                                 Defaults to None if not provided.
        
        Raises:
            ValueError: If values array is not one-dimensional.
            TypeError: If values is not a list or numpy array.
        """
        self.name = name
        
        # Input validation and conversion
        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError("Values must be a one-dimensional array.")
        elif isinstance(values, list):
            values = np.array(values, dtype=np.float64)
        else:
            raise TypeError("Values must be a list or a one-dimensional numpy array.")
        
        # Store values and generate unique identifier
        self._values = values
        self.id = self._values.tobytes()  # Unique identifier based on values

    @property
    def values(self):
        """
        Read-only access to policy coordinate values.
        
        This property provides a read-only view of the policy coordinates.
        The returned array cannot be modified, ensuring the immutability
        of policy values after creation. This is crucial for maintaining
        data integrity in electoral simulations where policy positions
        should remain constant throughout calculations.
        
        Returns:
            np.ndarray: Read-only view of policy coordinates.
                        Shape is (n_dimensions,) where n_dimensions is
                        typically 2 for electoral simulations.
                        Dtype is float64 for numerical precision.
        """
        read_only_view = self._values.view()
        read_only_view.setflags(write=False)  # Make the view read-only
        return read_only_view

    @values.setter
    def values(self):
        """
        Prevents modification of policy values after creation.
        
        This setter raises an AttributeError to enforce the immutability
        of policy values. Once a Policy object is created, its coordinates
        cannot be changed, ensuring data consistency throughout electoral
        simulations and preventing accidental modifications that could
        invalidate calculations.
        
        Raises:
            AttributeError: Always raised to prevent value modification.
                           Policy objects are immutable after creation.
        """
        raise AttributeError("Cannot modify the values of a Policy after it has been created.")
