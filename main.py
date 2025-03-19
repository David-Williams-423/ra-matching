# -------------------------- START IMPORTS -------------------------
# from rapidfuzz import fuzz

# from firebase_config import get_firestore_prefix
from algo_config import get_ilp_alpha, get_ilp_beta, get_max_rank

import pandas as pd
import sys
import os
import pulp
import traceback

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer

import requests

# -------------------------- END IMPORTS -------------------------


# -------------------------- START CSV INGESTION ------------------

if len(sys.argv) < 2:
    print("Usage: python main.py <input_file.csv>")
    sys.exit(1)

file_path = sys.argv[1]

try:
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)
    
    # Display basic information about the DataFrame
    print("CSV file successfully loaded into DataFrame")
    print(df.head())

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1)

# -------------------------- START CONFIG -------------------------

# nltk.download("stopwords")
# nltk.download("wordnet")

# Fetch Alpha parameter for the match probability
ILP_ALPHA = get_ilp_alpha()

# Fetch Beta parameter for the match probability
ILP_BETA = get_ilp_beta()

# Fetch the maximum amount of students that each faculty can recruit
MAX_RANK = get_max_rank()

# -------------------------- END CONFIG -------------------------

# ---------------------------- START FIRST ON-CALL MATHEMATICAL FUNCTIONS ----------------


# Probability calculation function for each match
def calculate_probability(faculty_rank_for_student, student_rank_for_faculty):
    faculty_contribution: float = (MAX_RANK - faculty_rank_for_student) * ILP_ALPHA
    student_contribution: float = (MAX_RANK - student_rank_for_faculty) * ILP_BETA
    raw_probability: float = faculty_contribution + student_contribution

    # Normalizing to ensure all values fall within 0 to 100.
    max_possible: float = (MAX_RANK - 1) * (ILP_ALPHA + ILP_BETA)
    return max((raw_probability / max_possible) * 100, 0)


def perform_ilp_matching(input_data: pd.DataFrame, faculty_prefs: dict) -> pd.DataFrame:
    """
    Solves the faculty-student matching problem allowing faculties to match with multiple students
    based on the length of their preference list in the faculty_prefs dictionary.

    Parameters:
        input_data (pd.DataFrame): The input DataFrame containing the following columns:
            - 'faculty_onyen': Faculty identifiers
            - 'student_onyen': Student identifiers
            - 'probability_of_match': Score/probability of the match
        faculty_prefs (dict): Dictionary where keys are faculty IDs and values are lists of student IDs
                              representing faculty preferences.

    Returns:
        pd.DataFrame: A DataFrame containing the optimal matches with columns:
            - 'faculty_onyen': Matched faculty identifiers
            - 'student_onyen': Matched student identifiers
            - 'probability_of_match': Probability of the match
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

    # Constraints: Each student can be matched with at most one faculty
    for student in input_data["student_onyen"].unique():
        problem += (
            pulp.lpSum(
                [
                    x[i]
                    for i in range(len(pairs))
                    if pairs[i]["student_onyen"] == student
                ]
            )
            <= 1,
            f"Student_Assignment_{student}",
        )

    # Constraints: Each faculty can be matched with up to their number of openings
    for faculty, prefs in faculty_prefs.items():
        num_openings = len(prefs)
        problem += (
            pulp.lpSum(
                [
                    x[i]
                    for i in range(len(pairs))
                    if pairs[i]["faculty_onyen"] == faculty
                ]
            )
            <= num_openings,
            f"Faculty_Openings_{faculty}",
        )

    # Solve the ILP problem
    problem.solve()

    # Extract the matches from the solution
    final_matching = [
        {
            "faculty_onyen": pairs[i]["faculty_onyen"],
            "student_onyen": pairs[i]["student_onyen"],
            "probability_of_match": pairs[i]["probability_of_match"],
        }
        for i in range(len(pairs))
        if pulp.value(x[i]) == 1
    ]

    # Return the final matching as a DataFrame
    return pd.DataFrame(final_matching)


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
