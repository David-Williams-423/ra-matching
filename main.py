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

    shell = MatchingShell(file_path_faculty, file_path_student)
    shell.cmdloop("\nRA/TA Matching Shell\n" +
                  f"Initial faculty weight: {shell.current_weight}\n" +
                  "Type 'help' for available commands")

    # try:
    #     # Read CSV file into DataFrame
    #     df_student = pd.read_csv(file_path_student)
    # except FileNotFoundError:
    #     print(f"Error: File '{file_path_student}' not found.")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     sys.exit(1)  
    # try:
    #     # Read CSV file into DataFrame
    #     df_faculty = pd.read_csv(file_path_faculty)
    # except FileNotFoundError:
    #     print(f"Error: File '{file_path_faculty}' not found.")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     sys.exit(1)

    # input_data, faculty_slots = process_preferences(df_student, df_faculty)

    # input_data, mandatory_matches, faculty_slots = assign_mandatory_matches(input_data, faculty_slots)

    # ilp_matches = perform_ilp_matching(input_data, faculty_slots)

    # combined_matches = pd.concat([mandatory_matches, ilp_matches], ignore_index=True)
    # combined_matches = combined_matches.sort_values('probability_of_match', ascending=False)

    # print(combined_matches)





if __name__ == "__main__": 
    main()