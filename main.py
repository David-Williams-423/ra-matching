# -------------------------- START IMPORTS -------------------------

from config import load_config

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
    global FACULTY_WEIGHT, NO_RANK_PENALTY, LOW_RANK_PENALTY

    if len(sys.argv) < 3:
        print("Usage: python main.py <student_file.csv> <faculty_file.csv> [<locking_file.csv>]")
        sys.exit(1)

    file_path_student = sys.argv[1]
    file_path_faculty = sys.argv[2]
    file_path_locking = None
    if (len(sys.argv)) > 3:
        file_path_locking = sys.argv[3]


    try:
        # Read CSV file into DataFrame
        df_student = pd.read_csv(file_path_student)
    except FileNotFoundError:
        print(f"Error: File '{file_path_student}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)  
    try:
        # Read CSV file into DataFrame
        df_faculty = pd.read_csv(file_path_faculty)
    except FileNotFoundError:
        print(f"Error: File '{file_path_faculty}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    if (file_path_locking is not None):
        try:
            # Read CSV file into DataFrame
            df_locking = pd.read_csv(file_path_locking)
        except FileNotFoundError:
            print(f"Error: File '{file_path_locking}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            sys.exit(1)

    input_data, faculty_slots = process_preferences(df_student, df_faculty)

    input_data, mandatory_matches, faculty_slots = assign_mandatory_matches(input_data, faculty_slots)

    ilp_matches = perform_ilp_matching(input_data, faculty_slots)

    combined_matches = pd.concat([mandatory_matches, ilp_matches], ignore_index=True)
    combined_matches = combined_matches.sort_values('probability_of_match', ascending=False)

    print(combined_matches)


# -------------------------- START CONFIG -------------------------

config = load_config()
FACULTY_WEIGHT = config["faculty_weight"]
LOW_RANK_PENALTY = config["low_rank_penalty"]
NO_RANK_PENALTY = config["no_rank_penalty"]

# -------------------------- END CONFIG -------------------------

# ---------------------------- START PREPROCESSING FUNCTIONS ----------------

# Probability calculation function for each match
def calculate_probability(student_rank, faculty_rank, method='normal'):
    # Calculate student rank score    
    student_rank_score = 1.0 - (student_rank - 1) * LOW_RANK_PENALTY if student_rank > 0 else 0
    
    # Calculate faculty rank score
    faculty_rank_score = 1.0 - (faculty_rank - 1) * LOW_RANK_PENALTY if faculty_rank > 0 else 0
    
    # Combine scores (weighted average)
    # Apply a penalty factor if either party didn't rank the other
    if student_rank <= 0 or faculty_rank <= 0:
        # Option 1: Use a multiplicative penalty
        return NO_RANK_PENALTY * ((faculty_rank_score * FACULTY_WEIGHT) + 
                                (student_rank_score * (1 - FACULTY_WEIGHT)))
    else:
        # Normal calculation for mutual rankings
        return (faculty_rank_score * FACULTY_WEIGHT) + (student_rank_score * (1 - FACULTY_WEIGHT))


def process_preferences(student_prefs_df: pd.DataFrame, faculty_prefs_df: pd.DataFrame):
    """
    Process the raw preference DataFrames into a comprehensive format for ILP matching.
    
    Parameters:
        student_prefs_df (pd.DataFrame): DataFrame containing student preferences
        faculty_prefs_df (pd.DataFrame): DataFrame containing faculty preferences
        
    Returns:
        tuple: (input_data, faculty_slots)
            - input_data: DataFrame with all possible faculty-student pairs and match probabilities
            - faculty_slots: Dictionary mapping faculty to their project slots
    """
    # Create a mapping of project names to their details
    faculty_projects = {}
    faculty_slots = {}
    
    # Extract faculty projects
    for _, row in faculty_prefs_df.iterrows():
        faculty_name = row['Full Name']
        
        # Process each project
        project_num = 1
        
        while f'Project #{project_num}' in row and not pd.isna(row[f'Project #{project_num}']):
            project_name = row[f'Project #{project_num}']
            full_project_identifier = f"{faculty_name} - {project_name}"
            
            # Handle slot count (careful with potential column name variations)
            slots_col = f'Number of Open Slots' if project_num == 1 else f'Number of Open Slots.{project_num-1}'
            slots = int(row[slots_col])
            
            # Extract student rankings for this project
            student_rankings = []
            for rank in range(1, 6):  # Assuming max 5 student rankings per project
                rank_col = f'Student Rank {rank}' if project_num == 1 else f'Student Rank {rank}.{project_num-1}'
                if rank_col in row and not pd.isna(row[rank_col]):
                    student_rankings.append(row[rank_col])
            
            # Store project details
            faculty_projects[full_project_identifier] = {
                'project_name': project_name,
                'slots': slots,
                'student_rankings': student_rankings,
                'faculty_name': faculty_name,
                'original_project_name': project_name
            }
            
            faculty_slots[full_project_identifier] = slots
            
            # Check for additional projects
            has_another_col = f'I have another project' if project_num == 1 else f'I have another project.{project_num-1}'
            has_another = row.get(has_another_col, False)
            if not has_another or pd.isna(has_another):
                break
                
            project_num += 1
    
    # Generate pairs for ALL students and projects
    pairs = []
    students = student_prefs_df['Full Name'].unique()
    
    for student_name in students:
        student_row = student_prefs_df[student_prefs_df['Full Name'] == student_name].iloc[0]
        
        # Get student's faculty preferences
        student_faculty_prefs = []
        for rank in range(1, 7):
            rank_col = f'Rank {rank}'
            if rank_col in student_row and not pd.isna(student_row[rank_col]):
                student_faculty_prefs.append(student_row[rank_col])

        # Generate match probabilities for ALL projects
        for project_identifier, project in faculty_projects.items():
            # Calculate base probability
            # First, check student's preference for this faculty
            student_rank = -1
            for rank, faculty_choice in enumerate(student_faculty_prefs, 1):
                if faculty_choice == project['project_name']:
                    student_rank = rank
                    break

            # Check faculty's ranking of this student
            faculty_rank = -1
            for i, ranked_student in enumerate(project['student_rankings'], 1):
                if ranked_student == student_name:
                    faculty_rank = i
                    break
        
            match_probability = calculate_probability(student_rank, faculty_rank)
            
            # Append pair information
            pairs.append({
                'faculty_project': project_identifier,
                'student_name': student_name,
                'probability_of_match': match_probability,
                'student_rank': student_rank,
                'faculty_rank': faculty_rank,
                'original_project_name': project['original_project_name'],
                'faculty_name': project['faculty_name']
            })
    
    return pd.DataFrame(pairs), faculty_slots


def assign_mandatory_matches(input_data: pd.DataFrame, faculty_slots: dict, locks: pd.DataFrame):
    """
    Identify and assign mandatory matches where both student and faculty 
    have each other as their first choice.
    
    Parameters:
    input_data (pd.DataFrame): DataFrame containing all possible student-faculty pairings
    faculty_slots (dict): Dictionary mapping faculty projects to number of open slots
    
    Returns:
    tuple: 
        - Modified input_data (DataFrame) with mandatory matches removed
        - Mandatory matches (DataFrame)
        - Updated faculty_slots dictionary
    """
    # Create a copy of faculty_slots to avoid modifying the original
    updated_faculty_slots = faculty_slots.copy()
    
    # Group the input data by student and faculty project
    grouped = input_data.groupby(['student_name', 'faculty_project'])
    
    # Find mandatory matches (where both student and faculty rank each other #1)
    mandatory_matches = []
    remaining_pairs = input_data.copy()
    
    # Iterate through unique student-faculty project combinations
    for (student, faculty_project), group in grouped:
        # Check if this is a mandatory match
        match_row = group.iloc[0]
        
        # Conditions for a mandatory match:
        # 1. Student rank is 1 (first choice)
        # 2. Faculty rank is 1 (first choice)
        if (match_row['student_rank'] == 1) and (match_row['faculty_rank'] == 1):
            # Verify there are still slots available for this project
            if updated_faculty_slots.get(faculty_project, 0) > 0:
                # Add to mandatory matches
                mandatory_matches.append({
                    'faculty_project': faculty_project,
                    'student_name': student,
                    'probability_of_match': match_row['probability_of_match'],
                    'student_rank': match_row['student_rank'],
                    'faculty_rank': match_row['faculty_rank'],
                    'original_project_name': match_row['original_project_name'],
                    'faculty_name': match_row['faculty_name']
                })
                
                # Reduce available slots for this project
                updated_faculty_slots[faculty_project] -= 1
                
                # Remove this match from remaining pairs
                remaining_pairs = remaining_pairs[
                    ~((remaining_pairs['student_name'] == student) | 
                      (remaining_pairs['faculty_project'] == faculty_project))
                ]
    
    # Convert mandatory matches to DataFrame
    mandatory_matches_df = pd.DataFrame(mandatory_matches)
    
    return remaining_pairs, mandatory_matches_df, updated_faculty_slots


# ---------------------------- END PREPROCESSING FUNCTIONS ----------------

# ---------------------------- START ILP FUNCTIONS ------------------------

def perform_ilp_matching(input_data: pd.DataFrame, faculty_slots: dict):
    """
    Solves the faculty-student matching problem using two separate preference DataFrames.
    
    Parameters:
        student_prefs_df (pd.DataFrame): DataFrame containing student preferences with columns:
            'Full Name', 'Rank 1', 'Rank 2', etc.
        faculty_prefs_df (pd.DataFrame): DataFrame containing faculty preferences with columns:
            'Full Name', 'Project #1', 'Number of Open Slots', 'Student Rank 1', etc.
            
    Returns:
        pd.DataFrame: A DataFrame containing the optimal matches with columns:
            - 'faculty_project': Matched faculty project
            - 'student_name': Matched student name
            - 'probability_of_match': Probability of the match
            - 'student_rank': The rank the student gave this faculty
            - 'faculty_rank': The rank the faculty gave this student
    """

    # Convert the input DataFrame to a list of dictionaries for easier access
    pairs = input_data.to_dict("records")

    # Initialize the ILP problem to maximize the objective
    problem = pulp.LpProblem("Faculty_Student_Matching", pulp.LpMaximize)

    # Define binary decision variables for each faculty-student pair
    x = pulp.LpVariable.dicts("match", (range(len(pairs))), cat="Binary")

    # Objective function: Maximize the weighted sum of probabilities of assigned matches
    problem += pulp.lpSum(
        [pairs[i]["probability_of_match"] * x[i] for i in range(len(pairs))]
    )

    # Constraints: Each student can be matched with at most one faculty project
    for student in input_data["student_name"].unique():
        problem += (
            pulp.lpSum(
                [
                    x[i]
                    for i in range(len(pairs))
                    if pairs[i]["student_name"] == student
                ]
            )
            <= 1,
            f"Student_Assignment_{student}",
        )

    # Constraints: Each faculty project can be matched with up to their number of openings
    for faculty_project, num_openings in faculty_slots.items():
        problem += (
            pulp.lpSum(
                [
                    x[i]
                    for i in range(len(pairs))
                    if pairs[i]["faculty_project"] == faculty_project
                ]
            )
            <= num_openings,
            f"Faculty_Openings_{faculty_project}",
        )

    # Solve the ILP problem
    problem.solve()

    # Check if an optimal solution was found
    if pulp.LpStatus[problem.status] != "Optimal":
        print(f"Warning: No optimal solution found. Status: {pulp.LpStatus[problem.status]}")
        return pd.DataFrame()  # Return empty DataFrame if no solution

    # Extract the matches from the solution
    final_matching = [
        {
            "faculty_project": pairs[i]["faculty_project"],
            "student_name": pairs[i]["student_name"],
            "probability_of_match": pairs[i]["probability_of_match"],
            "student_rank": pairs[i]["student_rank"],
            "faculty_rank": pairs[i]["faculty_rank"],
            "original_project_name": pairs[i]["original_project_name"],
            "faculty_name": pairs[i]["faculty_name"]
        }
        for i in range(len(pairs))
        if pulp.value(x[i]) == 1
    ]

    # Return the final matching as a DataFrame
    return pd.DataFrame(final_matching)

# ---------------------------- END ILP FUNCTIONS --------------------------


if __name__ == "__main__": 
    main()