"""
Test program details and conversation context
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
print("PROGRAM DETAILS & CONTEXT TEST")
print("="*60)

print("\nInitializing assistant...")
assistant = AIVoiceAssistant("university_dataset_advanced.json")

# Test 1: Program details
print("\n" + "="*60)
print("TEST 1: Tell me about B.Tech AI")
print("="*60)

start = time.time()
response = assistant.interact_with_llm("Tell me about B.Tech AI")
elapsed = time.time() - start

print(f"\nResponse ({elapsed:.2f}s):")
print(response)
print(f"\n{'✓ PASS' if elapsed < 2.0 else '⚠ SLOW'}: {elapsed:.2f}s")

# Test 2: Follow-up question (should remember B.Tech AI)
print("\n" + "="*60)
print("TEST 2: What about the fees? (Follow-up)")
print("="*60)

start = time.time()
response = assistant.interact_with_llm("What about the fees?")
elapsed = time.time() - start

print(f"\nResponse ({elapsed:.2f}s):")
print(response)
print(f"\n{'✓ PASS' if 'b.tech' in response.lower() or 'ai' in response.lower() else '✗ FAIL - Wrong context'}")

# Test 3: Another program
print("\n" + "="*60)
print("TEST 3: Tell me about MBA")
print("="*60)

start = time.time()
response = assistant.interact_with_llm("Tell me about MBA")
elapsed = time.time() - start

print(f"\nResponse ({elapsed:.2f}s):")
print(response[:300])
print(f"\n{'✓ PASS' if elapsed < 2.0 else '⚠ SLOW'}: {elapsed:.2f}s")

print("\n" + "="*60)
print("DONE")
print("="*60)
