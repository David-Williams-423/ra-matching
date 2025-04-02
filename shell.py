"""Shell implementation."""

import cmd
import sys
import pandas as pd
from algo_config import get_faculty_weight, set_faculty_weight
from main import (
    process_preferences,
    assign_mandatory_matches,
    perform_ilp_matching,
)


class MatchingShell(cmd.Cmd):
    """Interactive shell for RA/TA matching with live configuration."""

    prompt = '(match)'

    def __init__(self, faculty_file, student_file):
        """Initialize the shell with faculty and student data files."""
        super().__init__()
        self.faculty_file = faculty_file
        self.student_file = student_file
        self.current_weight = get_faculty_weight()
        self.original_faculty_slots = None
        self.load_initial_data()

    def load_initial_data(self):
        """Load initial data and perform initial matching."""
        try:
            self.df_student = pd.read_csv(self.student_file)
            self.df_faculty = pd.read_csv(self.faculty_file)
            self.process_data()
            print(
                f"Loaded {len(self.df_student)} students and "
                f"{len(self.df_faculty)} faculty."
            )
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            sys.exit(1)

    def process_data(self):
        """Re-run processing with current weights."""
        input_data, faculty_slots = process_preferences(self.df_student, self.df_faculty, self.current_weight)
        self.original_faculty_slots = faculty_slots.copy()
        
        input_data, self.mandatory_matches, updated_slots = assign_mandatory_matches(input_data, faculty_slots)
        self.ilp_matches = perform_ilp_matching(input_data, updated_slots)
        self.combined_matches = pd.concat([self.mandatory_matches, self.ilp_matches], ignore_index=True)
        self.combined_matches.sort_values('probability_of_match', ascending=False, inplace=True)

    def do_run_matching(self, arg):
        """Execute matching with the current configuration."""
        print("\nRunning matching algorithm...")
        self.process_data()
        print(f"Generated {len(self.matches)} matches.")
        print("Use 'show_matches' to view the results.")

    def do_change_weights(self, arg):
        """Adjust faculty/student preference weighting
        Usage: change_weights [0-1] (e.g., change_weights 0.5)]
        """
        try:
            new_weight = float(arg)
            if not 0 <= new_weight <= 1:
                raise ValueError("Weight must be between 0 and 1.")
        except ValueError as e:
            print(f"Invalid weight: {e}")
            return
    
        set_faculty_weight(new_weight)
        self.current_weight = new_weight
        print(f"\nWeights update - Faculty preference weight: {new_weight}")
        print(f"Run 'run_matching' to re-run the algorithm with new weights.")

    def do_show_matches(self, arg):
        """Display current matches.
        Usage: show_matches [--top N]
        """
        if self.combined_matches.empty:
            print("No matches calculated yet.")
            return
        
        # Parse optional args
        top_n = None
        if '--top' in arg:
            try:
                top_n = int(arg.split()[1])
                print(f"\nTop {top_n} matches:")
                print(self.combined_matches.head(top_n).to_string(index=False))
            except:
                print("Invalid format. Usage: show_matches [--top N]")
        else:
            print("\nCurrent matches:")
            print(self.combined_matches.to_string(index=False))

    def do_exit(self, arg):
        """Exit the shell."""
        print("Exiting...")
        return True
                

