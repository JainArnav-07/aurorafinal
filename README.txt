AURORA: Intelligent User Notification System
=============================================

ARCHITECTURE OVERVIEW
Three-stage pipeline orchestrates behavioral segmentation → message generation → personalized delivery.

STAGE 1: USER SEGMENTATION & PROFILING (task1.ipynb)
- Input: users.csv (user data), KB.md (company knowledge base)
- Processing:
  * StandardScaler normalizes user features
  * PCA reduces dimensionality across user engagement metrics
  * GaussianMixture (BIC-optimized) clusters users into behavioral segments
  * LLM (Ollama llama3.2) maps segments to Octalysis 8-Core Drives
- Output files:
  * users_segmented.csv (users + segment_id + lifecycle_stage)
  * segment_themes.json (segment profiles + Octalysis themes ranked by relevance)
  * company_north_star.json (inferred primary engagement metric)
  * feature_goal_map.json (app feature → user goal mappings)
  * allowed_tone_hook_matrix.json (communication constraints)

STAGE 2: MESSAGE & TIMING OPTIMIZATION (task2_timewondow.ipynb)
Four sub-tasks:

2a. Tone/Hook Assignment
- LLM selects 2-3 tones + 1-2 hooks per segment based on Octalysis theme
- Output: communication_themes.csv

2b. Template Generation
- Generates 5 push notification templates per (segment × lifecycle stage × Octalysis drive)
- Structured output: English + Hindi content + psychological hook + feature reference
- Output: message_templates.csv

2c. Timing Window Optimization
- GMM fits user preferred_hour distribution per segment (max 3 clusters)
- Maps cluster means to 6 time windows: early_morning, mid_morning, afternoon, late_afternoon, evening, night
- Calculates expected CTR + engagement_score per window
- Output: timing_recommendations.csv

2d. User Frequency Assignment
- Assigns daily_notification_limit based on activeness_score:
  * > 0.7: 7-9 notifs/day
  * 0.4-0.7: 5-6 notifs/day
  * < 0.4: 3-4 notifs/day
- Output: users_frequency_assigned.csv

STAGE 3: NOTIFICATION SCHEDULING & LEARNING (task3_users_notifications_generation.ipynb)
- Assigns message templates to each user based on:
  * Segment + lifecycle stage + optimal time window + daily frequency limit
- Weighted randomization selects templates to avoid fatigue
- Output: user_notification_schedule.csv + message assignments
- Delta reporting captures experiment results for A/B testing iterations

KEY MODELS
----------
LLM: ChatOllama (llama3.2, temperature=0)
  - Used for: Octalysis theme inference, tone/hook selection, template generation

ML: Scikit-learn
  - GaussianMixture (BIC criterion): Segment discovery + timing window detection
  - StandardScaler: Feature normalization
  - PCA: Dimensionality reduction for segment visualization

Data Flow: users/KB → segmentation → theme inference → template gen → timing → scheduling

RUN INSTRUCTIONS
----------------
Prerequisites:
  - Python 3.8+
  - pip packages: pandas, numpy, sklearn, langchain, langchain-ollama, pydantic
  - Ollama running locally (ollama serve)
  - llama3.2 model pulled (ollama pull llama3.2:latest)
  
Files required:
  - input/users.csv
  - input/KB.md
  - codebase/prompts/prompts.json

Execution:
  1. Navigate to codebase/
  2. Run task1.ipynb - Generates all segment profiles and themes
  3. Run task2.ipynb - Creates templates, timing windows, and frequency assignments
  4. Run task3.ipynb - Generates final user notification schedule
  
Outputs flow to output/ directory with timestamped backups in output/temp/

For iteration/learning loops, compare experiment_results.csv against previous iterations
stored in iteration_0_before_learning/ and iteration_1_after_learning/

CONFIGURATION NOTES
-------------------
- Adjust ollama_model variable for llama3.1 or other models
- GaussianMixture max_windows parameter controls timing granularity (default: 3)
- LLM temperature in task1 set to 0 (deterministic) for consistent themes
- Template generation uses temperature=0.3 for structured diversity
