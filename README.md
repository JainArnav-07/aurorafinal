# Aurora - Intelligent User Notification System

## Overview

Aurora is an AI-powered notification orchestration system designed to improve user engagement through intelligent segmentation, personalized content generation, timing optimization, and continuous feedback-based learning.

---

## Architecture

### Stage 1: Segmentation & Profiling

- User data preprocessing
- Missing value handling
- PCA-based activation score generation
- User segmentation into behavioral cohorts
- Knowledge extraction using RAG

### Stage 2: Message & Timing Intelligence

- Tone and hook assignment
- Personalized notification generation
- Octalysis-based gamification mapping
- GMM-based timing optimization
- Frequency assignment based on activation score

### Stage 3: Scheduling & Learning

- Notification schedule generation
- Experiment result ingestion
- Template performance evaluation
- Adaptive reweighting
- Continuous optimization loop

---

## Technology Stack

- Python
- Pandas
- Scikit-Learn
- ChromaDB
- Ollama (LLaMA 3.2)
- LangGraph

---

## Pipeline Flow

Raw User Data
↓
User Profiling
↓
Segmentation
↓
Template Generation
↓
Timing Optimization
↓
Notification Scheduling
↓
Experiment Feedback
↓
Adaptive Learning

---

## Repository Structure

- input/
- output/
- prompts/
- task1.ipynb
- task2_timewindow.ipynb
- task3_users_notifications.ipynb

---

## Features

- Behavioral User Segmentation
- Personalized Notification Generation
- Timing Window Optimization
- Feedback-Based Learning
- Adaptive Notification Scheduling

---

## Authors

Aurora Project Team