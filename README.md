# RA/TA/LA Matching Tool

## Introduction
An optimization system for automating graduate student job assignments using constrained optimization. Prioritizes:
- Client requirements through configurable weights
- Mutual preference alignment between students/faculty
- Project slot constraints
- Transparent probabilistic scoring

## Prerequisites
- **Python 3.7+**
- Required packages in `requirements.txt`
- Input CSV files formatted as specified below

### Installation

Build ra-matching from the source and install dependencies:

1. **Clone the repository:**

    ```bash
    ❯ git clone https://github.com/David-Williams-423/ra-matching
    ```

2. **Navigate to the project directory:**

    ```bash
    ❯ cd ra-matching
    ```

3. **Install the dependencies:**

    ```bash
    ❯ pip install -r requirements.txt
    ```

## How to Use

### 1. Configure Settings (Optional)
Edit `config.yaml` to adjust matching behavior:
```yaml
# faculty_weight: Weight given to faculty preferences in match calculations
# Range: 0.0 to 1.0
# - 0.0: Only student preferences matter
# - 0.5: Equal weight to student and faculty preferences
# - 1.0: Only faculty preferences matter
faculty_weight = 0.5  # 0 = student decides, 1 = faculty decides

# NO_RANK_PENALTY: Controls how much to reduce match probability when 
#   only one party includes the other in their ranking
# Range: 0.0 to 1.0
# - 0.0: One-sided preferences result in zero probability (most strict)
# - 0.5: One-sided preferences have their probability reduced by half (balanced)
# - 1.0: No penalty applied (same as original algorithm)
student_no_rank_penalty: 0.5
faculty_no_rank_penalty: 0.5

# LOW_RANK_PENALTY: Controls how much to reduce match probability when
#    a party ranks another lower than 1
# Range: 0.0 to 0.2
# - 0.0: A rank of 1 is treated the same as a rank of 5
# - 0.2: Rank of 1 given score of 1.0, rank of 2 given score of 0.8, etc.
low_rank_penalty: 0.15


# SIMILARITY_WEIGHT: Controls the influence of similarity between previous matches and new matches in the optimization process.
#    A higher weight prioritizes maintaining consistency with prior matchings.
# Range: 0.0 to 0.5
# - 0.0: No influence from prior matchings; optimization is based solely on current criteria.
# - 0.5: Strong emphasis on aligning with prior matchings, balancing with current criteria.
similarity_weight: 0.5


```

### 2. Prepare Input Files
**students.csv**
- These are the student preferences

Format:
```csv
Full Name,Rank 1,Rank 2,Rank 3,Rank 4,Rank 5,Rank 6
Alice Chen,"Machine Learning","NLP","Computer Vision","","",""
Bob Lee,"Robotics","HCI","","","",""
```

**faculty.csv**
- These are the faculty preferences

Format:
```csv
Full Name,Project #1,Number of Open Slots,Student Rank 1,Student Rank 2,Student Rank 3,Student Rank 4,Student Rank 5,I have another project
Dr. Smith,"NLP Research",2,"Alice Chen","Bob Lee","Charlie Brown","","",No
Dr. Jones,"Robot Vision",1,"Emma Wilson","","","","",Yes
```

**excluded_locked.csv**
- This is an optional file that contains the pairs of students and faculty that are to be locked or excluded.
	- Locked: This student, faculty, project combination will be paired together
	- Excluded: This student, faculty, project combination will not be paired together

Format:
```csv
Faculty Name,Project,Student Name,Locked,Excluded
Professor 3,"Renewable Energy Research Engineer",Samuel Garcia,true,false
Professor 1,"Machine Learning Research Scientist",Isaac Cohen,false,true
```

**previous_matching.csv**
- This is an optional file that contains the output of a previous matching result, in the case where a rerun is desired with minimal alterations to the previous run.

Format:
```csv
faculty_project,student_name,probability_of_match,student_rank,faculty_rank,original_project_name,faculty_name
Professor 4 - Autonomous Vehicle Research Assistant,Grace Hopper,1.0,1,1,Autonomous Vehicle Research Assistant,Professor 4
Professor 1 - AI Ethics Researcher,Lucas Bennett,1.0,1,1,AI Ethics Researcher,Professor 1
Professor 1 - Machine Learning Research Scientist,Olivia Chen,0.15,1,-1,Machine Learning Research Scientist,Professor 1
Professor 1 - Genomics Research Scientist,Priya Sharma,0.895,1,2,Genomics Research Scientist,Professor 1
Professor 2 - Molecular Biology Research Associate,Ethan Nguyen,0.895,1,2,Molecular Biology Research Associate,Professor 2
```

Note: This is the same format of the output file

### 3. Run the Program

```bash
python main.py <students.csv> <faculty.csv> [<excluded_locked.csv>] [<previous_matching.csv>]
```

<details> <summary><b>Function Descriptions</b></span></summary> <blockquote> <table style='width: 100%; border-collapse: collapse;'> <thead> <tr style='background-color: #f8f9fa;'> <th style='width: 30%; text-align: left; padding: 8px;'>Function Name</th> <th style='text-align: left; padding: 8px;'>Description</th> </tr> </thead> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>run_matching</b></td> <td style='padding: 8px;'>Executes the matching algorithm with the current configuration. Generates matches based on the input data and constraints. Outputs the number of matches generated.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>run_rematching</b></td> <td style='padding: 8px;'>Executes the rematching algorithm, incorporating results from a previous run. Useful for refining matches or addressing unmatched cases.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>change_faculty_weight</b></td> <td style='padding: 8px;'>Adjusts the faculty/student preference weighting. Usage: <code>change_faculty_weight [0-1]</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>change_low_rank_penalty</b></td> <td style='padding: 8px;'>Adjusts the penalty applied for lower-ranked preferences. Usage: <code>change_low_rank_penalty [0-1]</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>change_student_no_rank_penalty</b></td> <td style='padding: 8px;'>Modifies the penalty applied when a student has not ranked a project. Usage: <code>change_student_no_rank_penalty [0-1]</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>change_faculty_no_rank_penalty</b></td> <td style='padding: 8px;'>Modifies the penalty applied when a faculty member has not ranked a student. Usage: <code>change_faculty_no_rank_penalty [0-1]</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>show_matches</b></td> <td style='padding: 8px;'>Displays the matches generated by the algorithm. Can show all matches or the top N matches sorted by a selected field.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>change_sort</b></td> <td style='padding: 8px;'>Changes the field by which matches are sorted. Supports various flags such as <code>-f</code> (faculty_project), <code>-p</code> (probability_of_match), and more.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>show_config</b></td> <td style='padding: 8px;'>Displays the current configuration values, such as faculty weight, penalties, and similarity weight.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>change_similarity_weight</b></td> <td style='padding: 8px;'>Adjusts the similarity weight for matching. Usage: <code>change_similarity_weight [0-0.5]</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>show_locks_exclusions</b></td> <td style='padding: 8px;'>Displays the current locking file, detailing locked and excluded pairings.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>lock</b></td> <td style='padding: 8px;'>Adds a lock (mandatory pairing) to the locking file. Usage: <code>lock -f "Faculty Name" -p "Project Name" -s "Student Full Name"</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>exclude</b></td> <td style='padding: 8px;'>Adds an exclusion (disallowed pairing) to the locking file. Usage: <code>exclude -f "Faculty Name" -p "Project Name" -s "Student Full Name"</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>remove_lock</b></td> <td style='padding: 8px;'>Removes a lock from the locking file. Usage: <code>remove_lock -f "Faculty Name" -p "Project Name" -s "Student Full Name"</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>remove_exclusion</b></td> <td style='padding: 8px;'>Removes an exclusion from the locking file. Usage: <code>remove_exclusion -f "Faculty Name" -p "Project Name" -s "Student Full Name"</code>.</td> </tr> <tr style='border-bottom: 1px solid #eee;'> <td style='padding: 8px;'><b>return_csv</b></td> <td style='padding: 8px;'>Exports the current matches to a CSV file. Usage: <code>return_csv &lt;filename&gt;</code>.</td> </tr> <tr> <td style='padding: 8px;'><b>exit</b></td> <td style='padding: 8px;'>Exits the interactive matching shell.</td> </tr> </table> </blockquote> </details>

### 4. Understand Output
The system outputs a sorted list of matches with columns:

| Column | Description |
|--------|-------------|
| `faculty_project` | Faculty name + project identifier |
| `student_name` | Matched student |
| `probability_of_match` | Match quality score (0.0-1.0) |
| `student_rank` | Student's preference rank (0=unranked) |
| `faculty_rank` | Faculty's preference rank (0=unranked) |
| `original_project_name` | Original name of the faculty member's project | 
| `faculty_name` | Name of the faculty member | 

Example output:
```
               faculty_project                               student_name  probability_of_match  student_rank  faculty_rank
0  Professor 4 - Autonomous Vehicle Research Assistant       Grace Hopper            1.00             1             1
1  Professor 1 - AI Ethics Researcher                        Lucas Bennett           1.00             1             1
2  Professor 1 - Machine Learning Research Scientist         Olivia Chen             0.15             1            -1
```

## Data Requirements
**Student CSV Must Contain:**
- 1+ faculty/project rankings per student
- 6 maximum ranked preferences (columns `Rank 1`-`Rank 6`)

**Faculty CSV Must Contain:**
- 1-5 projects per faculty member
- Student rankings for each project
- Exact column names as shown in the example

## Algorithm Workflow
1. **Preprocess Inputs**
	- Calculate mutual preference probabilities.
	- Identify mandatory first-choice matches.
	- Process locking and excluding information:
	- Locked matches are enforced and cannot be altered.
	- Excluded matches are explicitly disallowed.
	- Optimize Remaining Matches

2. **Use Integer Linear Programming (ILP).**
	- Maximize: Σ(match_probability × assignment).
	- Constraints:
		- 1 match per student maximum.
		- Project slot limits.
		- Enforce locked matches and exclude disallowed matches from consideration.


3. **Handle Rematching (if required)**
	- Re-run the optimization for unmatched students or unfilled projects.
	- Adjust constraints dynamically based on prior results.

4. **Combine Results**
	- Merge locked matches, mandatory matches, and optimized matches.
	- Sort by match probability (highest first).

### Project Index

<details>
		<summary><b>File Summaries</b></summary>
		<blockquote>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/config.py'>config.py</a></b></td>
					<td style='padding: 8px;'>- Config.py` centralizes algorithm configuration parameters<br>- It loads settings from a YAML file, providing defaults and input validation to ensure parameter ranges are respected<br>- The module offers functions to load, save, and access individual configuration values, facilitating flexible parameter management within the broader application<br>- This ensures consistent and controlled access to crucial algorithm settings.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>- The LICENSE file specifies the projects open-source licensing terms<br>- It grants users broad permissions to use, modify, and distribute the software under the MIT License, a permissive license that minimizes liability for the copyright holder, David Williams<br>- This ensures legal clarity and facilitates community contribution and wider adoption of the project.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Requirements.txt` specifies the projects dependencies<br>- It ensures the projects successful execution by defining necessary versions of Pandas for data manipulation, PuLP for optimization, PyYAML for configuration file handling, and Argparse for command-line argument parsing<br>- These libraries provide the foundational tools for the application's core functionality.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/shell.py'>shell.py</a></b></td>
					<td style='padding: 8px;'>- The <code>shell.py</code> file provides an interactive command-line interface (CLI) for a Resident Advisor (RA)/Teaching Assistant (TA) matching system<br>- It allows users to load student and faculty data, configure matching preferences (likely using the <code>config</code> module), and perform the matching process (leveraging functions from the <code>utils</code> module, including an Integer Linear Programming (ILP) solver)<br>- The CLI facilitates interactive management of the matching process, potentially incorporating lock and exclusion constraints from a separate file<br>- The system appears to maintain and update matching results throughout the interactive session.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/config.yaml'>config.yaml</a></b></td>
					<td style='padding: 8px;'>- Config.yaml` defines weighting parameters for a ranking algorithm<br>- It specifies penalties for low ranks and weights for faculty and student input, influencing the overall ranking calculation within the larger project<br>- These parameters control the relative importance of different ranking factors, impacting the final ranked output<br>- The configuration facilitates adjustments to the ranking process without modifying core code.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/utils.py'>utils.py</a></b></td>
					<td style='padding: 8px;'>- Utils.py` provides utility functions for a student-faculty matching system<br>- It preprocesses preference data, calculating match probabilities based on ranking and applying penalties for missing rankings<br>- The module then performs an integer linear programming (ILP) optimization to find the optimal matching, considering locks, exclusions, and optionally, previous matchings to maximize overall match probability and similarity to prior assignments.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/test_matching.py'>test_matching.py</a></b></td>
					<td style='padding: 8px;'>- Tests validate the student-faculty project matching algorithm<br>- Unit tests cover probability calculation, preference processing, mandatory match assignment, and an integer linear programming (ILP) based optimization for final matching<br>- The tests ensure accurate probability computation, correct handling of preferences and constraints, and optimal assignment based on available slots and rankings.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/David-Williams-423/ra-matching/blob/master/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- The <code>main.py</code> script executes a RA/TA matching program<br>- It takes student and faculty data files as input, optionally incorporating locking and previous matching data<br>- The program uses a shell interface, driven by the <code>MatchingShell</code> class, allowing interactive matching and manipulation of the data based on a configurable faculty weight<br>- The script manages command-line arguments and program initialization.</td>
				</tr>
			</table>
		</blockquote>
	</details>
			</details>
		</blockquote>
</details>
