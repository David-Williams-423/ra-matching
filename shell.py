"""Shell implementation."""

import cmd
import sys
import pandas as pd
from utils import (
    process_preferences,
    assign_mandatory_matches,
    perform_ilp_matching,
    process_locks_exclusions
)
from config import (
    get_config_value,
    set_config_value,
)


class MatchingShell(cmd.Cmd):
    """Interactive shell for RA/TA matching with live configuration."""

    prompt = '(match)> '

    def __init__(self, student_file, faculty_file, locking_file=None):
        """Initialize the shell with faculty and student data files."""
        super().__init__()
        self.faculty_file = faculty_file
        self.student_file = student_file
        self.locking_file = locking_file
        self.current_weight = get_config_value('faculty_weight')
        self.original_faculty_slots = None
        self.load_initial_data()

    def load_initial_data(self):
        """Load initial data and perform initial matching."""
        try:
            # Read CSV file into DataFrame
            self.df_student = pd.read_csv(self.student_file)
        except FileNotFoundError:
            print(f"Error: File '{self.student_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            sys.exit(1)  
        try:
            # Read CSV file into DataFrame
            self.df_faculty = pd.read_csv(self.faculty_file)
        except FileNotFoundError:
            print(f"Error: File '{self.faculty_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            sys.exit(1)
        if (self.locking_file is not None):
            try:
                # Read CSV file into DataFrame
                self.df_locking = pd.read_csv(self.locking_file)
            except FileNotFoundError:
                print(f"Error: File '{self.locking_file}' not found.")
                sys.exit(1)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                sys.exit(1)

        self.process_data()
        print(
                f"Loaded {len(self.df_student)} students and "
                f"{len(self.df_faculty)} faculty."
            )

    def process_data(self):
        """Re-run processing with current weights."""
        input_data, faculty_slots = process_preferences(self.df_student, self.df_faculty)
        if self.locking_file is not None:
            locks, exclusions = process_locks_exclusions(self.df_locking)
        else:
            locks = None
            exclusions = None
        self.original_faculty_slots = faculty_slots.copy()
        
        input_data, self.mandatory_matches, updated_slots = assign_mandatory_matches(input_data, faculty_slots)
        self.ilp_matches = perform_ilp_matching(input_data, updated_slots, exclusions)
        self.combined_matches = pd.concat([self.mandatory_matches, self.ilp_matches], ignore_index=True)
        self.combined_matches.sort_values('probability_of_match', ascending=False)

    def do_run_matching(self, arg):
        """Execute matching with the current configuration."""
        print("\nRunning matching algorithm...")
        self.process_data()
        print(f"Generated {len(self.ilp_matches)} matches.")
        print("Use 'show_matches' to view the results.")

    def do_change_faculty_weight(self, arg):
        """Adjust faculty/student preference weighting
        Usage: change_faculty_weight [0-1] (e.g., change_faculty_weight 0.5)
        """
        try:
            new_weight = float(arg)
            if not 0 <= new_weight <= 1:
                raise ValueError("Weight must be between 0 and 1.")
        except ValueError as e:
            print(f"Invalid weight: {e}")
            print("Make sure you input the files in the correct order: python main.py <student_file> <faculty_file>")
            return
    
        set_config_value('faculty_weight', new_weight)
        self.current_weight = new_weight
        print(f"\nWeights update - Faculty preference weight: {new_weight}")
        print(f"Run 'run_matching' to re-run the algorithm with new weights.")

    def do_change_low_rank_penalty(self, arg):
        """Adjust low rank penalty
        Usage: change_low_rank_penalty [0-1] (e.g., change_low_rank_penalty 0.5)
        """
        try:
            new_penalty = float(arg)
            if not 0 <= new_penalty <= 1:
                raise ValueError("Penalty must be between 0 and 1.")
        except ValueError as e:
            print(f"Invalid penalty: {e}")
            return
    
        set_config_value('low_rank_penalty', new_penalty)
        print(f"\nLow rank penalty updated to: {new_penalty}")
        print(f"Run 'run_matching' to re-run the algorithm with new penalties.")

    def do_change_student_no_rank_penalty(self, arg):
        """Adjust student no rank penalty
        Usage: change_student_no_rank_penalty [0-1] (e.g., change_student_no_rank_penalty 0.5)
        """
        try:
            new_penalty = float(arg)
            if not 0 <= new_penalty <= 1:
                raise ValueError("Penalty must be between 0 and 1.")
        except ValueError as e:
            print(f"Invalid penalty: {e}")
            return
    
        set_config_value('no_rank_penalty', new_penalty)
        print(f"\nStudent no rank penalty updated to: {new_penalty}")
        print(f"Run 'run_matching' to re-run the algorithm with new penalties.")

    def do_change_faculty_no_rank_penalty(self, arg):
        """Adjust faculty no rank penalty
        Usage: change_faculty_no_rank_penalty [0-1] (e.g., change_faculty_no_rank_penalty 0.5)
        """
        try:
            new_penalty = float(arg)
            if not 0 <= new_penalty <= 1:
                raise ValueError("Penalty must be between 0 and 1.")
        except ValueError as e:
            print(f"Invalid penalty: {e}")
            return
    
        set_config_value('faculty_no_rank_penalty', new_penalty)
        print(f"\nFaculty no rank penalty updated to: {new_penalty}")
        print(f"Run 'run_matching' to re-run the algorithm with new penalties.")

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

    def do_return_csv(self, arg):
        """Export current matches to CSV.
        Usage: return_csv <filename>
        """
        if not arg:
            print("Please provide a filename.")
            return
        
        try:
            self.combined_matches.to_csv(arg, index=False)
            print(f"Matches exported to {arg}.")
        except Exception as e:
            print(f"Failed to export: {e}")

    def do_exit(self, arg):
        """Exit the shell."""
        print("Exiting...")
        return True
                

