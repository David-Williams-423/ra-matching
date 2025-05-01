# RA/TA/LA Matching Tool

## Introduction
An optimization system for automating graduate student job assignments using constrained optimization. Prioritizes:
- Client requirements through configurable weights
- Mutual preference alignment between students/faculty
- Project slot constraints
- Transparent probabilistic scoring

## Prerequisites
- **Python 3.7+**
- Required packages:
  ```bash
  pip install pandas pulp
  ```
- Input CSV files formatted as specified below

## Installing Dependencies
To install the required Python dependencies for this project, follow these steps:

Navigate to the project directory:

```bash
cd ra-matching```

Install the dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt```

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
These are the student preferences
***Format:***
```csv
Full Name,Rank 1,Rank 2,Rank 3,Rank 4,Rank 5,Rank 6
Alice Chen,"Machine Learning","NLP","Computer Vision","","",""
Bob Lee,"Robotics","HCI","","","",""
```

**faculty.csv**
These are the faculty preferences
***Format:***
```csv
Full Name,Project #1,Number of Open Slots,Student Rank 1,Student Rank 2,Student Rank 3,Student Rank 4,Student Rank 5,I have another project
Dr. Smith,"NLP Research",2,"Alice Chen","Bob Lee","Charlie Brown","","",No
Dr. Jones,"Robot Vision",1,"Emma Wilson","","","","",Yes
```

**excluded_locked.csv**
These is an optional file that contains the pairs of students and faculty that are to be locked or excluded.
Locked: This student, faculty, project combination will be paired together
Excluded: This student, faculty, project combination will not be paired together
***Format:***
```csv
Faculty Name,Project,Student Name,Locked,Excluded
Professor 3,"Renewable Energy Research Engineer",Samuel Garcia,true,false
Professor 1,"Machine Learning Research Scientist",Isaac Cohen,false,true
```

**previous_matching.csv**
These is an optional file that contains the output of a previous matching result, in the case where a rerun is desired with minimal alterations to the previous run.
***Format:***
```csv
faculty_project,student_name,probability_of_match,student_rank,faculty_rank,original_project_name,faculty_name
Professor 4 - Autonomous Vehicle Research Assistant,Grace Hopper,1.0,1,1,Autonomous Vehicle Research Assistant,Professor 4
Professor 1 - AI Ethics Researcher,Lucas Bennett,1.0,1,1,AI Ethics Researcher,Professor 1
Professor 1 - Machine Learning Research Scientist,Olivia Chen,0.15,1,-1,Machine Learning Research Scientist,Professor 1
Professor 1 - Genomics Research Scientist,Priya Sharma,0.895,1,2,Genomics Research Scientist,Professor 1
Professor 2 - Molecular Biology Research Associate,Ethan Nguyen,0.895,1,2,Molecular Biology Research Associate,Professor 2
```

Note: This is the same format of the output file

### 3. Run the Matching

```bash
python main.py <students.csv> <faculty.csv> [<excluded_locked.csv>] [<previous_matching.csv>]
```

### 4. Understand Output
The system outputs a sorted list of matches with columns:

| Column | Description |
|--------|-------------|
| `faculty_project` | Faculty name + project identifier |
| `student_name` | Matched student |
| `probability_of_match` | Match quality score (0.0-1.0) |
| `student_rank` | Student's preference rank (0=unranked) |
| `faculty_rank` | Faculty's preference rank (0=unranked) |
| 'original_project_name' | Original name of the faculty member's project | 
| 'faculty_name' | Name of the faculty member | 

Example output:
```
               faculty_project                     student_name  probability_of_match  student_rank  faculty_rank
0  Professor 4 - Autonomous Vehicle Research Assistant       Grace Hopper                  1.00             1             1
1                    Professor 1 - AI Ethics Researcher       Lucas Bennett                  1.00             1             1
2       Professor 1 - Machine Learning Research Scientist        Olivia Chen                  0.15             1            -1
```

### 5. Save Results (Optional)
Redirect output to a file:
```bash
python main.py students.csv faculty.csv > matches.csv
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
   - Calculate mutual preference probabilities
   - Identify mandatory first-choice matches

2. **Optimize Remaining Matches**  
   - Use Integer Linear Programming (ILP)
   - Maximize: Σ(match_probability × assignment)
   - Constraints: 1 match/student max, project slot limits

3. **Combine Results**  
   - Mandatory matches + optimized matches
   - Sort by match probability (highest first)
