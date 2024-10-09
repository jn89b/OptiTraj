from abc import ABC, abstractmethod
from typing import Dict

def validate_limits(limits_dict:Dict, limit_type="control"):
    """
    Validate the structure and values of the limits dictionary.
    
    Parameters:
        limits_dict (dict): Dictionary containing state or control limits.
        limit_type (str): Type of limits being validated ('control' or 'state').
        
    Raises:
        ValueError: If the dictionary structure or values are invalid.
    """
    required_keys = ['min', 'max']

    for variable, limits in limits_dict.items():
        # Check that the limits are in a dictionary
        if not isinstance(limits, dict):
            raise ValueError(f"{limit_type.capitalize()} '{variable}' should be a dictionary with 'min' and 'max' keys.")
        
        # Check that 'min' and 'max' keys exist
        for key in required_keys:
            if key not in limits:
                raise ValueError(f"{limit_type.capitalize()} '{variable}' is missing '{key}' key.")
            
            # Check that 'min' and 'max' are numbers (either int or float)
            if not isinstance(limits[key], (int, float)):
                raise ValueError(f"{limit_type.capitalize()} '{variable}' has a non-numeric '{key}' value: {limits[key]}")
        
        # Check that 'min' is less than or equal to 'max'
        if limits['min'] > limits['max']:
            raise ValueError(f"{limit_type.capitalize()} '{variable}' has 'min' greater than 'max': {limits['min']} > {limits['max']}")

    print(f"{limit_type.capitalize()} limits are valid.")


class Limits():
    """Concrete class for defining control limits."""

    def __init__(self, limits):
        """Initialize with a dictionary of control limits."""
        self.limits = limits

    def get_min(self, name: str):
        """Return the minimum limit for the given control."""
        return self.limits[name]['min']

    def get_max(self, name: str):
        """Return the maximum limit for the given control."""
        return self.limits[name]['max']