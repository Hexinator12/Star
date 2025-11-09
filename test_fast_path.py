"""
Test the fast path directly
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
print("FAST PATH TEST")
print("="*60)

print("\nInitializing assistant...")
assistant = AIVoiceAssistant("university_dataset_advanced.json")

print("\n" + "="*60)
print("TEST 1: List all programs")
print("="*60)

start = time.time()
response = assistant.interact_with_llm("List all programs")
elapsed = time.time() - start

print(f"\nResponse ({elapsed:.2f}s):")
print(response[:500])  # First 500 chars
print(f"\n{'✓ PASS' if elapsed < 1.0 else '✗ FAIL'}: {elapsed:.2f}s")

print("\n" + "="*60)
print("TEST 2: How many programs?")
print("="*60)

start = time.time()
response = assistant.interact_with_llm("How many programs?")
elapsed = time.time() - start

print(f"\nResponse ({elapsed:.2f}s):")
print(response)
print(f"\n{'✓ PASS' if elapsed < 1.0 else '✗ FAIL'}: {elapsed:.2f}s")

print("\n" + "="*60)
print("DONE")
print("="*60)
