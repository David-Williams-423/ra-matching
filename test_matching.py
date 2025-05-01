import pytest
import pandas as pd

# Import the functions to test.
from utils import (
    calculate_probability,
    process_preferences,
    assign_mandatory_matches,
    perform_ilp_matching,
    FACULTY_WEIGHT
)

# ------------------------------
# Tests for calculate_probability
# ------------------------------
def test_calculate_probability_perfect_match():
    # When both student and faculty have rank 1, both scores should be 1.
    # Expected probability = 1 * FACULTY_WEIGHT + 1 * (1 - FACULTY_WEIGHT) = 1.
    prob = calculate_probability(1, 1)
    assert pytest.approx(prob, rel=1e-2) == 1.0

def test_calculate_probability_second_choice():
    # For rank 2, each score should be 1 - 0.15 = 0.85.
    # Expected probability = 0.85 * FACULTY_WEIGHT + 0.85 * (1 - FACULTY_WEIGHT) = 0.85.
    prob = calculate_probability(2, 2)
    assert pytest.approx(prob, rel=1e-2) == 0.85

def test_calculate_probability_student_rank_zero():
    # If student rank is 0 (or non-positive), student score becomes 0.
    # For faculty_rank 3, faculty score = 1 - (3-1)*0.15 = 0.7.
    # Expected probability = 0.7 * FACULTY_WEIGHT + 0 * (1 - FACULTY_WEIGHT)
    prob = calculate_probability(0, 2)
    expected = 0.2975
    assert pytest.approx(prob, rel=1e-2) == expected

def test_calculate_probability_negative_rank():
    # Negative rankings are treated as non-positive, so score becomes 0.
    # For example, student_rank = -1 (score 0) and faculty_rank = 2 (score = 0.85)
    prob = calculate_probability(2, -1)
    expected = 0.1275  # since faculty score=0
    assert pytest.approx(prob, rel=1e-2) == expected

# ------------------------------
# Tests for process_preferences
# ------------------------------
def test_process_preferences_basic():
    # Create a simple student DataFrame with one student
    student_data = {
        "Full Name": ["Alice"],
        "Rank 1": ["Project A"]
    }
    student_df = pd.DataFrame(student_data)
    
    # Create a simple faculty DataFrame with one faculty having one project.
    faculty_data = {
        "Full Name": ["Prof. Smith"],
        "Project #1": ["Project A"],
        "Number of Open Slots": [2],
        "Student Rank 1": ["Alice"],
        "I have another project": [None]  # No additional project indicator
    }
    faculty_df = pd.DataFrame(faculty_data)
    
    input_data, faculty_slots = process_preferences(student_df, faculty_df)
    
    # There should be one match pair created.
    assert not input_data.empty
    row = input_data.iloc[0]
    # The faculty project identifier is constructed as "Faculty Name - Project Name"
    assert row['student_name'] == "Alice"
    assert row['faculty_project'] == "Prof. Smith - Project A"
    
    # Check that faculty_slots dictionary has the correct mapping and value.
    assert faculty_slots.get("Prof. Smith - Project A") == 2

# ------------------------------
# Tests for assign_mandatory_matches
# ------------------------------
def test_assign_mandatory_matches_mandatory_case():
    # Create an input_data DataFrame simulating a mandatory match where both ranks are 1.
    data = [
        {
            "faculty_project": "Prof. Brown - Project B",
            "student_name": "Bob",
            "probability_of_match": 1.0,
            "student_rank": 1,
            "faculty_rank": 1,
            "original_project_name": "Project B",
            "faculty_name": "Prof. Brown"
        },
        # Include an extra entry for the same project with a different student.
        {
            "faculty_project": "Prof. Brown - Project B",
            "student_name": "Charlie",
            "probability_of_match": 0.8,
            "student_rank": 2,
            "faculty_rank": 3,
            "original_project_name": "Project B",
            "faculty_name": "Prof. Brown"
        }
    ]
    input_df = pd.DataFrame(data)
    # Set available slots for the project to 1.
    faculty_slots = {"Prof. Brown - Project B": 1}
    
    remaining, mandatory, updated_slots = assign_mandatory_matches(input_df, faculty_slots)
    
    # The mandatory match list should include Bob.
    assert not mandatory.empty
    assert "Bob" in mandatory['student_name'].values
    # The available slots should have reduced for the project.
    assert updated_slots["Prof. Brown - Project B"] == 0
    # Ensure that Bob's match (or any match with the same student or project)
    # is removed from the remaining pairs.
    assert not any(remaining['student_name'] == "Bob")

# ------------------------------
# Tests for perform_ilp_matching
# ------------------------------
def test_perform_ilp_matching_single_student():
    # Create a simple input_data DataFrame where one student appears in two pairs.
    data = [
        {
            "faculty_project": "Prof. White - Project X",
            "student_name": "Dana",
            "probability_of_match": 0.9,
            "student_rank": 1,
            "faculty_rank": 2,
            "original_project_name": "Project X",
            "faculty_name": "Prof. White"
        },
        {
            "faculty_project": "Prof. Green - Project Y",
            "student_name": "Dana",
            "probability_of_match": 0.8,
            "student_rank": 2,
            "faculty_rank": 1,
            "original_project_name": "Project Y",
            "faculty_name": "Prof. Green"
        }
    ]
    input_df = pd.DataFrame(data)
    # Each project has one available slot.
    faculty_slots = {
        "Prof. White - Project X": 1,
        "Prof. Green - Project Y": 1
    }
    
    matches_df = perform_ilp_matching(input_df, faculty_slots)
    
    # Because each student can only be matched with one project,
    # we expect exactly one match for student Dana.
    assert len(matches_df) == 1
    chosen_match = matches_df.iloc[0]
    # The match should be the one with the higher probability (0.9).
    assert chosen_match['faculty_project'] == "Prof. White - Project X"
    assert chosen_match['student_name'] == "Dana"