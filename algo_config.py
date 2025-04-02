# Alpha is the penalty applied to the student's rankings for the faculty
# The higher alpha is, the more faculty's preferences matter more than students'
ILP_ALPHA = 3

# Beta is the penalty applied to the faculty's rankings for the student
# The higher beta is, the more students' preferences matter than faculty's
ILP_BETA = 1

"""
In general, it is a good idea to
"""

# MAX_RANK is a normalization parameter, indicating the maximum number
# of preferences that can be given during a matching cycle.
# The lower MAX_RANK is, the more sensitive the probability function becomes
# when considering ranks. Since, it's impractical that a single faculty would recruit
# more than ten students in a single cycle, 10 is a good trade-off between
# Sensitivity and Practicality
MAX_RANK = 10

# FACULTY_WEIGHT is the weighting of faculty preferences vs student
# This should be in the range [0,1]
# A value of 1 indicates that the faculty holds all of the consideration
# and a value of 0 indicates that the student holds all.
# A value of 0.5 is neutral
FACULTY_WEIGHT = 0.5


def get_ilp_alpha():
    """Get the ILP alpha value."""
    return ILP_ALPHA


def get_ilp_beta():
    """Get the ILP beta value."""
    return ILP_BETA


def get_max_rank():
    """Get the maximum rank value."""
    return MAX_RANK


def get_faculty_weight():
    """Get the faculty weight for matching."""
    return FACULTY_WEIGHT


def set_faculty_weight(new_weight):
    """Set the faculty weight for matching."""
    global FACULTY_WEIGHT
    if not 0 <= new_weight <= 1:
        raise ValueError("Weight must be between 0 and 1.")
    FACULTY_WEIGHT = new_weight
