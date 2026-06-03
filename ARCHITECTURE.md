# Aurora Architecture

## Problem

Traditional notification systems send generic messages at fixed times.

Aurora improves engagement by combining user segmentation, personalized content generation, timing optimization, and feedback-driven learning.

---

## Stage 1: User Segmentation

Input:
- User demographics
- Activity data
- Streak information
- Engagement signals

Processing:
- Missing value handling
- PCA-based activation scoring
- Rule-based cohort segmentation

Output:
- User segments (S1-S9)

---

## Stage 2: Message & Timing Intelligence

### Message Generation

- Octalysis-based motivation mapping
- Tone assignment
- Hook assignment
- Personalized template creation

### Timing Optimization

- Gaussian Mixture Models (GMM)
- Preferred hour clustering
- Optimal notification window selection

---

## Stage 3: Scheduling & Learning

### Iteration 0

Generate initial notification schedules.

### Iteration 1

Use experiment feedback:

- CTR
- Engagement
- Uninstall rate

to update notification strategy.

---

## Learning Loop

GOOD Template → Promote

NEUTRAL Template → Continue Testing

BAD Template → Suppress

---

## Technologies

- Python
- Pandas
- Scikit-Learn
- ChromaDB
- LangGraph
- Ollama (Llama 3.2)