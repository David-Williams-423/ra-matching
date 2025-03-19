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


def get_ilp_alpha():
    return ILP_ALPHA


def get_ilp_beta():
    return ILP_BETA


def get_max_rank():
    return MAX_RANK