AURORA: Intelligent User Notification System
=============================================

ARCHITECTURE OVERVIEW
Three-stage pipeline orchestrates behavioral segmentation → message generation → personalized delivery.

PROJECT STRUCTURE
-----------------
aurora_final/
├── README.txt
├── experiment_results.csv          # A/B test results used for iteration learning
├── codebase/
│   ├── .env                        # Ollama connection config (web vs local)
│   ├── instructions.txt
│   ├── task1.ipynb                 # Stage 1: Segmentation & KB extraction
│   ├── task2_timewondow.ipynb      # Stage 2: Templates & timing optimization
│   ├── task3_users_notifications_generation.ipynb  # Stage 3: Scheduling & learning
│   ├── input/
│   │   ├── users.csv               # Raw user data
│   │   ├── KB.md                   # Company knowledge base
│   │   ├── KB_v2.md                # Updated knowledge base
│   │   └── KB.pdf
│   ├── prompts/
│   │   └── prompts.json            # LLM prompt templates
│   └── output/                     # All generated artifacts
│       ├── users_imputed.csv
│       ├── users_propensity_generated.csv
│       ├── users_segmented.csv
│       ├── users_frequency_assigned.csv
│       ├── company_north_star.json
│       ├── allowed_tone_hook_matrix.json
│       ├── feature_goal_map.json
│       ├── segment_lifecycle_goals.csv
│       ├── segment_themes.json
│       ├── communication_themes.csv
│       ├── message_templates.csv
│       ├── message_templates_scored.csv
│       ├── message_templates_weighted_pool.csv
│       ├── timing_recommendations.csv
│       ├── user_schedules_final.csv        # Iteration 0 schedule
│       ├── user_schedules_final_iter2.csv  # Iteration 1 schedule
│       └── learning_delta_report.csv
├── iteration_0_before_learning/    # Snapshot of all outputs before experiment feedback
│   ├── user_segments.csv
│   ├── segment_goals.csv
│   ├── company_north_star.json
│   ├── allowed_tone_hook_matrix.json
│   ├── feature_goal_map.json
│   ├── communication_themes.csv
│   ├── message_templates.csv
│   ├── timing_recommendations.csv
│   └── user_notification_schedule.csv
└── iteration_1_after_learning/     # Updated outputs after applying experiment feedback
    ├── message_templates.csv
    ├── timing_recommendations.csv
    ├── user_notification_schedule.csv
    └── learning_delta_report.csv

STAGE 1: USER SEGMENTATION & PROFILING (task1.ipynb)
- Input: input/users.csv, input/KB.md
- Processing:
  * Missing values imputed (numeric column means)
  * Sigmoid-normalized propensity scores: gamification, ai_tutor, leaderboard
  * PCA-derived activation_score (sessions, exercises, streak, notif_open_rate, motivation)
  * Rule-based segmentation: dominant_propensity × activation_level → 9 named segments (S1–S9)
  * RAG pipeline chunks KB → Chroma vector store → retrieves context per extraction
  * LLM (Ollama llama3.2) extracts: north star metric, tone/hook matrix, feature-goal mappings
  * LLM generates Octalysis theme profiles and segment-level goals for all lifecycle stages
- Output files (codebase/output/):
  * users_imputed.csv
  * users_propensity_generated.csv
  * users_segmented.csv
  * company_north_star.json
  * allowed_tone_hook_matrix.json
  * feature_goal_map.json
  * segment_lifecycle_goals.csv
  * segment_themes.json
- Also mirrored to: iteration_0_before_learning/

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
- GMM fits user preferred_hour distribution per segment (max 3 clusters, BIC-selected)
- Maps cluster means to 6 time windows: early_morning, mid_morning, afternoon,
  late_afternoon, evening, night
- Calculates expected CTR + engagement_score per window
- Output: timing_recommendations.csv

2d. User Frequency Assignment
- Assigns daily_notification_limit based on activation_score:
  * > 0.7  → 8 notifs/day
  * 0.4–0.7 → 5 notifs/day
  * < 0.4  → 3 notifs/day
- Output: users_frequency_assigned.csv

STAGE 3: NOTIFICATION SCHEDULING & LEARNING (task3_users_notifications_generation.ipynb)
Three sub-tasks:

3a. Iteration 0 — Schedule Generation
- Merges users, timing windows, and templates into a wide-format daily schedule
- Assigns up to 9 notification slots per user (time, template_id, channel)
- Output: user_schedules_final.csv

3b. Iteration 1 — Experiment-Informed Update
- Reads experiment_results.csv to score templates (GOOD / NEUTRAL / BAD)
- Epsilon-greedy weighting: GOOD=100, NEUTRAL=1, BAD=0 (suppressed)
- Updates timing windows via composite score (0.5×CTR + 0.3×engagement - 0.2×uninstall)
  with softmax redistribution (T=0.05) and iterative pruning of low-share windows
- Caps daily notifications to 2/day for segments with uninstall rate > 2%
- Output: message_templates_scored.csv, message_templates_weighted_pool.csv,
          user_schedules_final_iter2.csv
- Mirrored to: iteration_1_after_learning/

3c. Delta Reporter
- Compares iteration 0 vs iteration 1 across templates, timing windows, and frequency limits
- Logs entity-level changes with metric triggers and before/after values
- Output: learning_delta_report.csv

KEY MODELS
----------
LLM: ChatOllama (llama3.2, temperature=0 for extraction, 0.3 for template generation)
  - Used for: KB extraction, Octalysis theme inference, tone/hook selection, template generation
  - Connection: configured via .env (web or local Ollama)

ML: Scikit-learn
  - GaussianMixture (BIC criterion): timing window detection per segment
  - PCA: activation score derivation

Data Flow:
  users.csv + KB.md
    → imputation → propensity scores → segmentation
    → RAG-based KB extraction → segment themes & goals
    → tone/hook assignment → template generation
    → GMM timing → frequency assignment
    → schedule generation → experiment feedback → weighted re-scheduling

RUN INSTRUCTIONS
----------------
Prerequisites:
  - Python 3.8+
  - pip packages: pandas, numpy, scikit-learn, langchain, langchain-ollama,
                  langchain-community, langchain-text-splitters, langgraph,
                  pydantic, chromadb, python-dotenv
  - Ollama accessible (local or via ngrok tunnel)
  - llama3.2 model pulled: ollama pull llama3.2:latest

Configuration (codebase/.env):
  OLLAMA_MODE=web          # "web" uses ngrok URL, "local" uses 127.0.0.1:11434
  OLLAMA_LOCAL_URL=http://127.0.0.1:11434
  OLLAMA_WEB_URL=https://unsympathizing-censorian-buena.ngrok-free.dev

Files required:
  - codebase/input/users.csv
  - codebase/input/KB.md
  - codebase/prompts/prompts.json

Execution order:
  1. Edit codebase/.env — set OLLAMA_MODE to "web" or "local"
  2. Run task1.ipynb        — generates segmentation, KB outputs, segment themes & goals
  3. Run task2_timewondow.ipynb — generates templates, timing windows, frequency assignments
  4. Run task3_users_notifications_generation.ipynb — generates schedules, runs learning loop

All outputs are written to codebase/output/.
Iteration snapshots are saved to iteration_0_before_learning/ and iteration_1_after_learning/.

NOTE ON ITERATION 1 OUTPUT
--------------------------
Since experiment_results.csv (A/B test feedback data) is not provided in this submission,
the iteration_1_after_learning/ folder is intentionally left empty and no Iteration 1
outputs have been pre-generated. This is expected — the code is fully functional.

To compute Iteration 1, simply place your experiment_results.csv in the project root
(home) directory and re-run task3_users_notifications_generation.ipynb. The pipeline will
automatically read the file, apply epsilon-greedy template weighting, update timing
windows, and populate iteration_1_after_learning/ with the updated schedule and delta
report.

CONFIGURATION NOTES
-------------------
- Switch between web and local Ollama by editing OLLAMA_MODE in codebase/.env
- Adjust ollama_model variable in each notebook for different model versions
- GaussianMixture max_windows controls timing granularity (default: 3)
- LLM temperature: 0 for deterministic extraction, 0.3 for template diversity
- Softmax temperature (SOFTMAX_TEMPERATURE=0.05) controls exploitation vs. exploration
  in timing window selection; lower = more aggressive exploitation of top windows
- Uninstall rate cap threshold (UNINSTALL_RATE_THRESHOLD=0.02) and cap limit
  _users_notifications_generation.ipynb

  GITHUB REPOSITORY
  -----------------
  https://github.com/Aurora-PS-IITG/aurorafinal
  