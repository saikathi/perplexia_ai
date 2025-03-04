import os
import sys
import argparse

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Perplexia AI Assistant')
parser.add_argument('--mode', type=str, choices=['part1', 'part2', 'part3'], 
                    default='part1', help='Which part of Week 1 to run')
args = parser.parse_args()

# Import and run the app
from perplexia_ai.app import create_demo

if __name__ == "__main__":
    demo = create_demo(args.mode)
    demo.launch()
