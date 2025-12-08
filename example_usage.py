#!/usr/bin/env python3
"""
Example usage of the integrated LLM-based multi-task execution system
"""

import os
import subprocess

# Example 1: Navigate only (no LLM, hardcoded)
print("Example 1: Navigate to a location (hardcoded plan)")
print("Command: python src/main.py")
print()

# Example 2: Navigate with LLM planning
print("Example 2: Navigate using LLM planning")
print("Command: python src/main.py --goal 'navigate to the kitchen' --use-llm")
print("Requires: OPENAI_API_KEY environment variable")
print()

# Example 3: Navigate and pick (LLM)
print("Example 3: Navigate and pick up an object")
print("Command: python src/main.py --goal 'go to the dining room and pick up the cup on the table' --use-llm")
print()

# Example 4: Pick with specific algorithm
print("Example 4: Use SAC for arm reaching")
print("Command: python src/main.py --goal 'navigate to drawer and open it' --use-llm --arm-algorithm SAC --arm-model logs/simple_arm/realistic_sac_20251207_062739/final_sac.zip")
print()

# Example 5: With video recording
print("Example 5: Record video of execution")
print("Command: python src/main.py --goal 'navigate to kitchen' --save-video")
print()

print("="*70)
print("QUICK START (without LLM):")
print("="*70)
print("python src/main.py")
print()

print("="*70)
print("WITH LLM (requires OpenAI API key):")
print("="*70)
print("export OPENAI_API_KEY='your-key-here'")
print("python src/main.py --goal 'navigate to kitchen and pick up the plate' --use-llm")
