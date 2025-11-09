"""
Test the recommendation system
"""

import sys
import io
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from AIVoiceAssistant_new import AIVoiceAssistant
import time

print("="*60)
print("RECOMMENDATION SYSTEM TEST")
print("="*60)

print("\nInitializing assistant...")
assistant = AIVoiceAssistant("university_dataset_advanced.json")

test_queries = [
    "Which program is best if I want to go in IT field?",
    "I want to work in AI, which program should I choose?",
    "Best program for business career?",
    "What program is good for design?",
    "I'm interested in robotics, which program?",
    "Which program should I choose?",  # General
]

for query in test_queries:
    print("\n" + "="*60)
    print(f"Query: {query}")
    print("="*60)
    
    start = time.time()
    response = assistant.interact_with_llm(query)
    elapsed = time.time() - start
    
    print(f"\nResponse ({elapsed:.2f}s):")
    print(response)
    print(f"\n{'✓ PASS' if elapsed < 1.0 else '⚠ SLOW'}: {elapsed:.2f}s")

print("\n" + "="*60)
print("DONE")
print("="*60)
