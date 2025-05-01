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
    set_config_value,
)


class MatchingShell(cmd.Cmd):
    """Interactive shell for RA/TA matching with live configuration."""

    prompt = '(match)> '

    def __init__(self, student_file, faculty_file, locking_file=None, previous_file=None):
        """Initialize the shell with faculty and student data files."""
        super().__init__()
        self.faculty_file = faculty_file
        self.student_file = student_file
        self.locking_file = locking_file
        self.original_faculty_slots = None
        self.previous_file = previous_file
        self.combined_matches = None
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
        if (self.previous_file is not None):
            try:
                # Read CSV file into DataFrame
                self.df_previous = pd.read_csv(self.previous_file)
            except FileNotFoundError:
                print(f"Error: File '{self.previous_file}' not found.")
                sys.exit(1)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                sys.exit(1)
        else:
            self.df_previous = None

        if (self.df_previous is not None):
            self.combined_matches = self.df_previous

        # self.process_data()
        print(
                f"Loaded {len(self.df_student)} students and "
                f"{len(self.df_faculty)} faculty."
            )

    def process_data(self, rematch):
        """Re-run processing with current weights."""
        input_data, faculty_slots = process_preferences(self.df_student, self.df_faculty)
        if self.locking_file is not None:
            locks, exclusions = process_locks_exclusions(self.df_locking)
        else:
            locks = None
            exclusions = None
        self.original_faculty_slots = faculty_slots.copy()
        
        input_data, self.mandatory_matches, updated_slots = assign_mandatory_matches(input_data, faculty_slots, locks)
        if (rematch):
            ilp_matches = perform_ilp_matching(input_data, updated_slots, exclusions, self.combined_matches)
        else:
            ilp_matches = perform_ilp_matching(input_data, updated_slots, exclusions)
        self.combined_matches = pd.concat([self.mandatory_matches, ilp_matches], ignore_index=True)
        self.combined_matches.sort_values('probability_of_match', ascending=False)

    def do_run_matching(self, arg):
        """Execute matching with the current configuration."""
        print("\nRunning matching algorithm...")
        self.process_data(rematch=False)
        print(f"Generated {len(self.combined_matches)} matches.")
        print("Use 'show_matches' to view the results.")

    def do_run_rematching(self, arg):
        """Execute rematching with current configuration and previous run"""
        print("\nRunning rematching algorithm...")
        self.process_data(rematch=True)
        print(f"Generated {len(self.combined_matches)} matches.")
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
        if self.combined_matches is None or self.combined_matches.empty:
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

    def do_show_locks_exclusions(self, arg):
        """Display current locking file.
        Usage: show_locks_exclusions
        """
        if self.locking_file is None:
            print("No locking file provided.")
            return
        
        try:
            df_locking = pd.read_csv(self.locking_file)
            print("\nCurrent locking file:")
            print(df_locking.to_string(index=False))
        except Exception as e:
            print(f"Failed to read locking file: {e}")
            return
    
    def do_lock(self, arg):
        """Set locking file.
        Usage: lock <faculty_name> <project_name> <student_name> [<filename>]
        """
        if not arg:
            print("Please provide the following arguments: <faculty_name> <project_name> <student_name> [<filename>]")
            return
        arg = arg.split()
        if self.locking_file is None:
            # Check for the optional filename argument
            if len(arg) < 3:
                print("Please provide a filename.")
                return

            
        try:
            self.locking_file = arg[3] if len(arg) == 4 else self.locking_file
            self.df_locking = pd.read_csv(self.locking_file)
            print(f"Locking file set to {arg}.")
        except FileNotFoundError:
            print(f"Error: File '{arg}' not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        # Add a row to the locking file csv
        new_row = {
            "Faculty Name": arg[0],
            "Project": arg[1],
            "Student Name": arg[2],
            "Locked": True,
            "Excluded": False,
        }
        if self.df_locking is not None:
            self.df_locking = pd.concat([self.df_locking, pd.DataFrame([new_row])], ignore_index=True)
        else:
            self.df_locking = pd.DataFrame([new_row])
        self.df_locking.to_csv(self.locking_file, index=False)
        print(f"Locking file updated with {new_row}.")
        print(f"Run 'run_matching' to re-run the algorithm with new locks.")

    

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
                

