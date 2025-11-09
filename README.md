# Simple Habitat-Lab Reinforcement Learning Project

This project implements a basic reinforcement learning agent using Habitat-Lab for navigation tasks.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `agent.py`: Implementation of the RL agent
  - `environment.py`: Habitat environment setup
  - `train.py`: Main training script
- `configs/`
  - `habitat_config.yaml`: Habitat environment configuration

## Usage

To train the agent:
```bash
python src/train.py
```

## Requirements

See `requirements.txt` for the list of Python dependencies.