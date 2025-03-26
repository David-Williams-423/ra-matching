# -------------------------- START IMPORTS -------------------------
# from rapidfuzz import fuzz

# from firebase_config import get_firestore_prefix
from algo_config import get_ilp_alpha, get_ilp_beta, get_max_rank, get_faculty_weight

import pandas as pd

# Display all rows
pd.set_option('display.max_rows', None)

# # Display all columns (optional)
# pd.set_option('display.max_columns', None)


import sys
import os
import pulp
import traceback
import pprint

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer

import requests

# -------------------------- END IMPORTS -------------------------


# -------------------------- MAIN FUNCTION ------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <student_file.csv> <faculty_file.csv>")
        sys.exit(1)

    file_path_student = sys.argv[1]
    file_path_faculty = sys.argv[2]

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

    input_data, faculty_slots = process_preferences(df_student, df_faculty)

    input_data, mandatory_matches, faculty_slots = assign_mandatory_matches(input_data, faculty_slots)

    ilp_matches = perform_ilp_matching(input_data, faculty_slots)

    combined_matches = pd.concat([mandatory_matches, ilp_matches], ignore_index=True)
    combined_matches = combined_matches.sort_values('probability_of_match', ascending=False)

    print(combined_matches)


# -------------------------- START CONFIG -------------------------

# nltk.download("stopwords")
# nltk.download("wordnet")

# Fetch Alpha parameter for the match probability
ILP_ALPHA = get_ilp_alpha()

# Fetch Beta parameter for the match probability
ILP_BETA = get_ilp_beta()

# Fetch the maximum amount of students that each faculty can recruit
MAX_RANK = get_max_rank()

FACULTY_WEIGHT = get_faculty_weight()

# -------------------------- END CONFIG -------------------------

# ---------------------------- START PREPROCESSING FUNCTIONS ----------------

# Probability calculation function for each match
def calculate_probability(student_rank, faculty_rank):
    # Calculate student rank score
    student_rank_score = 1.0 - (student_rank - 1) * 0.15 if student_rank > 0 else 0
    
    # Calculate faculty rank score
    faculty_rank_score = 1.0 - (faculty_rank - 1) * 0.15 if faculty_rank > 0 else 0
    
    # Combine scores (weighted average)
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
            
            # Calculate student rank score
            student_rank_score = 1.0 - (student_rank - 1) * 0.15 if student_rank > 0 else 0
            
            # Calculate faculty rank score
            faculty_rank_score = 1.0 - (faculty_rank - 1) * 0.15 if faculty_rank > 0 else 0
            
            # Combine scores (weighted average)
            match_probability = (faculty_rank_score * FACULTY_WEIGHT) + (student_rank_score * (1 - FACULTY_WEIGHT))

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


def assign_mandatory_matches(input_data: pd.DataFrame, faculty_slots: dict):
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


# ============================ PREVIOUS TEAM'S CODE =======================

# ---------------------------- END FIRST ON-CALL MATHEMATICAL FUNCTIONS ------------------

# --------------------------- START FIRST ON-CALL HELPER FUNCTIONS -----------------------------------


# Generate preference dictionary from the student documents
def generate_student_preferences(student_list):
    student_preferences = {}
    for student in student_list:
        student_preferences[student["onyen"]] = student["prefProfessors"]

    return student_preferences


# Generate preference dictionary by matching faculty documents with sorted
def generate_faculty_preferences(faculty_list, CURRENT_MATCHING):
    faculty_preferences = {}

    collection_ref = firestore_path_to_reference(
        FIRESTORE_PREFIX + "/matching/" + CURRENT_MATCHING + "/sorted"
    )
    try:
        for faculty in faculty_list:
            faculty_onyen = faculty["onyen"]

            faculty_doc_ref = collection_ref.document(faculty_onyen)

            faculty_doc = faculty_doc_ref.get()

            if faculty_doc.exists:
                faculty_preferences[faculty_onyen] = faculty_doc.to_dict()["finalized"]
            else:
                print(
                    f"Unable to find entry for faculty with onyen {faculty_onyen} in the sorted collection."
                )

        return faculty_preferences
    except Exception as e:
        print(f"Error generating faculty preferences: {str(e)}")
        return {}


# Generate master dataframe from the faculty and student preferences
def generate_master_df(faculty_preferences, student_preferences):

    df_columns = [
        "faculty_onyen",
        "student_onyen",
        "faculty_rank_for_student",
        "student_rank_for_faculty",
        "probability_of_match",
    ]

    # Initialize the master DataFrame
    master_df = pd.DataFrame(columns=df_columns)

    # Iterate through faculty preferences to generate matches
    for faculty_onyen, faculty_prefs in faculty_preferences.items():
        for student_onyen in faculty_prefs:
            # Check if the student has ranked the faculty
            if faculty_onyen in student_preferences.get(student_onyen, []):
                new_row = {
                    "faculty_onyen": faculty_onyen,
                    "student_onyen": student_onyen,
                }
                master_df = pd.concat(
                    [master_df, pd.DataFrame([new_row])], ignore_index=True
                )

    # Iterate through student preferences to generate matches
    for student_onyen, student_prefs in student_preferences.items():
        for faculty_onyen in student_prefs:
            # Check if the faculty has ranked the student
            if student_onyen in faculty_preferences.get(faculty_onyen, []):
                new_row = {
                    "faculty_onyen": faculty_onyen,
                    "student_onyen": student_onyen,
                }
                master_df = pd.concat(
                    [master_df, pd.DataFrame([new_row])], ignore_index=True
                )

    # Remove duplicate entries if any
    master_df.drop_duplicates(inplace=True, ignore_index=True)

    # Assign ranks for each match
    for index, row in master_df.iterrows():
        faculty_onyen = row["faculty_onyen"]
        student_onyen = row["student_onyen"]

        # Find faculty rank for the student
        faculty_rank_for_student = None
        if faculty_onyen in faculty_preferences:
            try:
                faculty_rank_for_student = (
                    faculty_preferences[faculty_onyen].index(student_onyen) + 1
                )
            except ValueError:
                faculty_rank_for_student = None

        # Find student rank for the faculty
        student_rank_for_faculty = None
        if student_onyen in student_preferences:
            try:
                student_rank_for_faculty = (
                    student_preferences[student_onyen].index(faculty_onyen) + 1
                )
            except ValueError:
                student_rank_for_faculty = None

        probability_of_match = calculate_probability(
            faculty_rank_for_student, student_rank_for_faculty
        )

        # Update the DataFrame
        master_df.at[index, "faculty_rank_for_student"] = faculty_rank_for_student
        master_df.at[index, "student_rank_for_faculty"] = student_rank_for_faculty
        master_df.at[index, "probability_of_match"] = probability_of_match

    return master_df


# Extract the names of students who were not matched by the algorithm
def extract_unmatched_students(student_preferences, matched_df):

    all_students = set(student_preferences.keys())

    matched_students = set(matched_df["student_onyen"])

    unmatched_students = all_students - matched_students
    return unmatched_students



# ---------------------------- END FIRST ON-CALL HELPER FUNCTIONS ----------------------------------


# --------------------------- START SECOND ON-CALL MATHEMATICAL FUNCTIONS -----------------------------------


def fuzzy_match_faculty_students(faculty_interests, student_interests, threshold=75):
    results = {}

    # Preprocess faculty and student interests
    faculty_interests = {
        faculty: preprocess_text(" ".join(f_interests))
        for faculty, f_interests in faculty_interests.items()
    }
    student_interests = {
        student: preprocess_text(s_interests)
        for student, s_interests in student_interests.items()
    }

    for faculty, f_interests_str in faculty_interests.items():
        best_matches = []

        for student, s_interests_str in student_interests.items():
            # Compare faculty interests with student interests using fuzz.partial_ratio
            score = fuzz.partial_ratio(f_interests_str, s_interests_str)

            if score >= threshold:  # Only consider matches above the threshold
                best_matches.append((student, score))

        # Sort matches by score in descending order
        best_matches.sort(key=lambda x: x[1], reverse=True)
        results[faculty] = best_matches

    return results


# ---------------------------- END SECOND ON-CALL MATHEMATICAL FUNCTIONS ------------------------------------

# --------------------------- START SECOND ON-CALL HELPER FUNCTIONS -----------------------------------


def preprocess_text(text):
    text = text.lower()

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in text.split() if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


def generate_faculty_interests(faculty_list):
    faculty_interests = {}
    # for faculty in faculty_list:
    #     faculty_interests[faculty["onyen"]] = {
    #         "positionsOpen": faculty["positionsOpen"],
    #         "researchInterests": ", ".join(faculty["researchInterests"]),
    #     }
    for faculty in faculty_list:
        faculty_interests[faculty["onyen"]] = (", ".join(faculty["researchInterests"]),)

    return faculty_interests


def generate_student_interests(student_list):
    student_interests = {}
    # for student in student_list:
    #     student_interests[student["onyen"]] = {
    #         "description": student["description"],
    #         "researchInterests": student["researchInterests"],
    #         "resumeLink": student["resumeLink"],
    #         "videoLink": student["videoLink"],
    #     }
    for student in student_list:
        student_interests[student["onyen"]] = (
            student["description"] + " " + ", ".join(student["researchInterests"])
        )

    return student_interests


# ---------------------------- END SECOND ON-CALL HELPER FUNCTIONS ------------------------------------


if __name__ == "__main__": 
    main()