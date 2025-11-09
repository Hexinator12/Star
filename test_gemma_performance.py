"""
Test performance with Gemma 2B model
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from AIVoiceAssistant_new import AIVoiceAssistant
import time

print("="*60)
print("GEMMA 2B PERFORMANCE TEST")
print("="*60)

print("\nInitializing assistant with Gemma 2B...")
start_init = time.time()
assistant = AIVoiceAssistant("university_dataset_advanced.json")
init_time = time.time() - start_init
print(f"✓ Initialized in {init_time:.2f}s")

test_cases = [
    {
        "name": "Fast Path - List Programs",
        "query": "List all programs",
        "expected_time": 0.5,
        "type": "fast"
    },
    {
        "name": "Fast Path - Recommendation",
        "query": "Which program is best for IT?",
        "expected_time": 0.5,
        "type": "fast"
    },
    {
        "name": "Fast Path - Program Details",
        "query": "Tell me about MBA",
        "expected_time": 0.5,
        "type": "fast"
    },
    {
        "name": "LLM Path - Specific Query",
        "query": "What is the eligibility for B.Tech Data Science?",
        "expected_time": 10.0,
        "type": "llm"
    },
    {
        "name": "LLM Path - Fees Query",
        "query": "What are the fees for MBA in Finance?",
        "expected_time": 10.0,
        "type": "llm"
    }
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}/{len(test_cases)}: {test['name']}")
    print(f"Query: '{test['query']}'")
    print(f"Expected: < {test['expected_time']}s")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        response = assistant.interact_with_llm(test['query'])
        elapsed = time.time() - start
        
        # Show response preview
        preview = response[:200] + "..." if len(response) > 200 else response
        print(f"\nResponse ({elapsed:.2f}s):")
        print(preview)
        
        # Evaluate
        if elapsed < test['expected_time']:
            print(f"\n✅ PASS: {elapsed:.2f}s < {test['expected_time']}s")
            status = "PASS"
        else:
            print(f"\n⚠️ SLOW: {elapsed:.2f}s >= {test['expected_time']}s")
            status = "SLOW"
        
        results.append({
            "test": test['name'],
            "time": elapsed,
            "expected": test['expected_time'],
            "status": status
        })
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        results.append({
            "test": test['name'],
            "time": -1,
            "expected": test['expected_time'],
            "status": "ERROR"
        })

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

fast_tests = [r for r in results if r['expected'] <= 0.5]
llm_tests = [r for r in results if r['expected'] > 0.5]

print(f"\n📊 Fast Path Tests ({len(fast_tests)}):")
for r in fast_tests:
    status_icon = "✅" if r['status'] == "PASS" else "❌"
    print(f"  {status_icon} {r['test']}: {r['time']:.2f}s")

print(f"\n🤖 LLM Path Tests ({len(llm_tests)}):")
for r in llm_tests:
    status_icon = "✅" if r['status'] == "PASS" else ("⚠️" if r['status'] == "SLOW" else "❌")
    print(f"  {status_icon} {r['test']}: {r['time']:.2f}s")

# Overall stats
passed = len([r for r in results if r['status'] == "PASS"])
total = len(results)
avg_fast = sum(r['time'] for r in fast_tests if r['time'] > 0) / len(fast_tests) if fast_tests else 0
avg_llm = sum(r['time'] for r in llm_tests if r['time'] > 0) / len(llm_tests) if llm_tests else 0

print(f"\n📈 Overall Performance:")
print(f"  Tests Passed: {passed}/{total}")
print(f"  Avg Fast Path Time: {avg_fast:.2f}s")
print(f"  Avg LLM Path Time: {avg_llm:.2f}s")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! System is production-ready!")
elif passed >= total * 0.8:
    print("\n✅ Most tests passed. System is ready for deployment.")
else:
    print("\n⚠️ Some tests failed. Review configuration.")

print("\n" + "="*60)
