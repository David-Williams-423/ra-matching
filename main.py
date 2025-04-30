# -------------------------- START IMPORTS -------------------------

from algo_config import get_faculty_weight
from shell import MatchingShell

import pandas as pd

# Display all rows
pd.set_option('display.max_rows', None)

# # Display all columns (optional)
# pd.set_option('display.max_columns', None)

import sys
import pulp

# -------------------------- END IMPORTS -------------------------


# -------------------------- MAIN FUNCTION ------------------

def main():
    """Main function to run the RA/TA matching shell."""
    if len(sys.argv) != 3:
        print("Usage: python main.py <student_file.csv> <faculty_file.csv>")
        sys.exit(1)

    file_path_student = sys.argv[1]
    file_path_faculty = sys.argv[2]

    shell = MatchingShell(file_path_student, file_path_faculty)
    shell.cmdloop("\nRA/TA Matching Shell\n" +
                  f"Initial faculty weight: {shell.current_weight}\n" +
                  "Type 'help' for available commands")





if __name__ == "__main__": 
    main()