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

## How to Use

### 1. Configure Settings (Optional)
Edit `algoconfig.py` to adjust matching behavior:
```python
# Weight between faculty/student preferences (0-1)
FACULTY_WEIGHT = 0.5  # 0 = student decides, 1 = faculty decides
```

### 2. Prepare Input Files
**students.csv Format:**
```csv
Full Name,Rank 1,Rank 2,Rank 3,Rank 4,Rank 5,Rank 6
Alice Chen,"Machine Learning","NLP","Computer Vision","","",""
Bob Lee,"Robotics","HCI","","","",""
```

**faculty.csv Format:**
```csv
Full Name,Project #1,Number of Open Slots,Student Rank 1,Student Rank 2,Student Rank 3,Student Rank 4,Student Rank 5,I have another project
Dr. Smith,"NLP Research",2,"Alice Chen","Bob Lee","Charlie Brown","","",No
Dr. Jones,"Robot Vision",1,"Emma Wilson","","","","",Yes
```

### 3. Run the Matching
```bash
python main.py students.csv faculty.csv
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

Example output:
```
               faculty_project  student_name  probability_of_match  student_rank  faculty_rank
0       Dr. Smith - NLP Research    Alice Chen                  0.95             1             1
1      Dr. Jones - Robot Vision   Emma Wilson                  0.90             1             1
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
