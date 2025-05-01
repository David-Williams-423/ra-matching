import pandas as pd
from config import get_config_value, set_config_value

import sys
import pulp

# -------------------------- START CONFIG -------------------------

FACULTY_WEIGHT = get_config_value("faculty_weight")
LOW_RANK_PENALTY = get_config_value("low_rank_penalty")
STUDENT_NO_RANK_PENALTY = get_config_value("student_no_rank_penalty")
FACULTY_NO_RANK_PENALTY = get_config_value("faculty_no_rank_penalty")
SIMILARITY_WEIGHT = get_config_value("similarity_weight")

def run_config():
    global FACULTY_WEIGHT, LOW_RANK_PENALTY, STUDENT_NO_RANK_PENALTY, FACULTY_NO_RANK_PENALTY, SIMILARITY_WEIGHT
    FACULTY_WEIGHT = get_config_value("faculty_weight")
    LOW_RANK_PENALTY = get_config_value("low_rank_penalty")
    STUDENT_NO_RANK_PENALTY = get_config_value("student_no_rank_penalty")
    FACULTY_NO_RANK_PENALTY = get_config_value("faculty_no_rank_penalty")
    SIMILARITY_WEIGHT = get_config_value("similarity_weight")

# -------------------------- END CONFIG -------------------------

# ---------------------------- START PREPROCESSING FUNCTIONS ----------------

# Probability calculation function for each match
def calculate_probability(student_rank, faculty_rank, method='normal'):
    run_config()
    # Calculate student rank score    
    student_rank_score = 1.0 - (student_rank - 1) * LOW_RANK_PENALTY if student_rank > 0 else 0
    
    # Calculate faculty rank score
    faculty_rank_score = 1.0 - (faculty_rank - 1) * LOW_RANK_PENALTY if faculty_rank > 0 else 0
    
    # Combine scores (weighted average)
    # Apply a penalty factor if either party didn't rank the other
    if student_rank <= 0 and faculty_rank <= 0:
        return 0.0
    elif student_rank <= 0:
        return STUDENT_NO_RANK_PENALTY * ((faculty_rank_score * FACULTY_WEIGHT) + 
                                        (student_rank_score * (1 - FACULTY_WEIGHT)))
    elif faculty_rank <= 0:
        return FACULTY_NO_RANK_PENALTY * ((faculty_rank_score * FACULTY_WEIGHT) + 
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

def process_locks_exclusions(locking_df: pd.DataFrame):
    """
    Process a DataFrame with columns "Faculty Project", "Student Name", "Locked", "Excluded"
    and extract lists of locks and exclusions.
    
    Args:
        locking_df (pd.DataFrame): DataFrame with assignment data
        
    Returns:
        Tuple containing:
            - List of locks (tuples of (project, student))
            - List of exclusions (tuples of (project, student))
    """
    # Validate column names
    required_columns = ["Faculty Name", "Project", "Student Name", "Locked", "Excluded"]
    if not all(col in locking_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in locking_df.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Check for rows with both Locked and Excluded as True
    invalid_rows = locking_df[(locking_df["Locked"] == True) & (locking_df["Excluded"] == True)]
    if not invalid_rows.empty:
        conflicting_rows = invalid_rows[["Faculty Project", "Student Name"]].values.tolist()
        raise ValueError(f"Found {len(invalid_rows)} rows with both Locked and Excluded as True: {conflicting_rows}")
    
    # Extract locks
    locks = []
    locked_rows = locking_df[locking_df["Locked"] == True]
    for _, row in locked_rows.iterrows():
        locks.append((f'{row["Faculty Name"]} - {row["Project"]}', row["Student Name"]))
    
    # Extract exclusions
    exclusions = []
    excluded_rows = locking_df[locking_df["Excluded"] == True]
    for _, row in excluded_rows.iterrows():
        exclusions.append((f'{row["Faculty Name"]} - {row["Project"]}', row["Student Name"]))
    
    return locks, exclusions

def assign_mandatory_matches(input_data: pd.DataFrame, faculty_slots: dict, locks: list = None):
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
        # 1. Student-Faculty Pairing in the Locked List
        # OR
        # 2a. Student rank is 1 (first choice)
        # 2b. Faculty rank is 1 (first choice)
        locked_pair = False
        if locks and len(locks) > 0:
            for (locked_faculty_project, locked_student) in locks:
                if locked_faculty_project == faculty_project and locked_student == student:
                    locked_pair = True


        if locked_pair or ((match_row['student_rank'] == 1) and (match_row['faculty_rank'] == 1)):
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

def perform_ilp_matching(input_data: pd.DataFrame, faculty_slots: dict,
                    exclusions: list = None, previous: pd.DataFrame = None):
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
    run_config()

    # Convert the input DataFrame to a list of dictionaries for easier access
    pairs = input_data.to_dict("records")

    # Initialize the ILP problem to maximize the objective
    problem = pulp.LpProblem("Faculty_Student_Matching", pulp.LpMaximize)

    # Define binary decision variables for each faculty-student pair
    x = pulp.LpVariable.dicts("match", (range(len(pairs))), cat="Binary")

    # Define the base probability maximization component
    probability_component = pulp.lpSum(
        [pairs[i]["probability_of_match"] * x[i] for i in range(len(pairs))]
    )

    # Add similarity component if there are previous matchings
    if previous is not None:
        # Convert previous matchings to a set for fast lookup
        previous_matches = set([(row["faculty_project"], row["student_name"]) 
                            for _, row in previous.iterrows()])
        
        # Define similarity component
        similarity_component = pulp.lpSum(
            [x[i] for i in range(len(pairs)) 
            if (pairs[i]["faculty_project"], pairs[i]["student_name"]) in previous_matches]
        )
        
        # Combined objective with weights
        problem += (1 - SIMILARITY_WEIGHT) * probability_component + SIMILARITY_WEIGHT * similarity_component
    else:
        # Just use probability component if no previous matchings
        problem += probability_component

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

    # Add constraints for exclusions if provided
    if exclusions and len(exclusions) > 0:
        for i in range(len(pairs)):
            faculty_project = pairs[i]["faculty_project"]
            student_name = pairs[i]["student_name"]
            
            # If this pair is in the exclusions list, force its x variable to be 0
            if (faculty_project, student_name) in exclusions:
                problem += (
                    x[i] == 0,
                    f"Exclusion_{faculty_project}_{student_name}"
                )

    # Solve the ILP problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

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