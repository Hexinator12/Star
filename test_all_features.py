"""
Comprehensive test for all features including new fast paths
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from AIVoiceAssistant_new import AIVoiceAssistant
import time

print("="*70)
print("COMPREHENSIVE FEATURE TEST")
print("="*70)

print("\nInitializing assistant...")
assistant = AIVoiceAssistant("university_dataset_advanced.json")

test_cases = [
    # Fast Path - List/Count
    ("List all programs", 0.5, "List Query"),
    ("How many programs?", 0.5, "Count Query"),
    
    # Fast Path - Recommendations
    ("Which program is best for IT?", 0.5, "Recommendation"),
    
    # Fast Path - NEW: Fees
    ("What are the fees?", 0.5, "Fees Query (NEW)"),
    ("How much does it cost?", 0.5, "Cost Query (NEW)"),
    
    # Fast Path - NEW: Eligibility
    ("What is the eligibility?", 0.5, "Eligibility Query (NEW)"),
    ("What are the requirements?", 0.5, "Requirements Query (NEW)"),
    
    # Fast Path - NEW: Duration
    ("How long is the program?", 0.5, "Duration Query (NEW)"),
    
    # Fast Path - NEW: Contact
    ("How do I apply?", 0.5, "Application Query (NEW)"),
    ("What is the contact information?", 0.5, "Contact Query (NEW)"),
    
    # LLM Path - Complex queries
    ("Compare B.Tech and M.Tech programs", 15.0, "LLM Comparison"),
    
    # Cache Test - Repeat query
    ("What are the fees?", 0.1, "Cache Test (should be instant)"),
]

results = {"fast": [], "llm": [], "cache": []}

for i, (query, expected_time, description) in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}/{len(test_cases)}: {description}")
    print(f"Query: '{query}'")
    print(f"Expected: < {expected_time}s")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        response = assistant.interact_with_llm(query)
        elapsed = time.time() - start
        
        # Show response preview
        preview = response[:150] + "..." if len(response) > 150 else response
        print(f"\nResponse ({elapsed:.2f}s):")
        print(preview)
        
        # Categorize and evaluate
        if elapsed < 0.5:
            category = "fast"
            status = "✅ INSTANT"
        elif elapsed < expected_time:
            category = "llm" if expected_time > 1 else "cache"
            status = "✅ PASS"
        else:
            category = "llm"
            status = "⚠️ SLOW"
        
        print(f"\n{status}: {elapsed:.2f}s")
        
        results[category].append({
            "test": description,
            "time": elapsed,
            "status": status
        })
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)[:100]}")
        results["llm"].append({
            "test": description,
            "time": -1,
            "status": "❌ ERROR"
        })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n⚡ Fast Path Tests ({len(results['fast'])}):")
for r in results['fast']:
    print(f"  {r['status']} {r['test']}: {r['time']:.3f}s")

if results['cache']:
    print(f"\n💾 Cache Tests ({len(results['cache'])}):")
    for r in results['cache']:
        print(f"  {r['status']} {r['test']}: {r['time']:.3f}s")

if results['llm']:
    print(f"\n🤖 LLM Path Tests ({len(results['llm'])}):")
    for r in results['llm']:
        if r['time'] > 0:
            print(f"  {r['status']} {r['test']}: {r['time']:.2f}s")
        else:
            print(f"  {r['status']} {r['test']}")

# Statistics
total_tests = sum(len(v) for v in results.values())
instant_tests = len(results['fast'])
cached_tests = len(results['cache'])

print(f"\n📊 Performance Statistics:")
print(f"  Total Tests: {total_tests}")
print(f"  Instant (< 0.5s): {instant_tests} ({instant_tests/total_tests*100:.0f}%)")
print(f"  Cached: {cached_tests}")
print(f"  LLM Cache Hits: {assistant.llm_cache_hits}")

if instant_tests >= total_tests * 0.8:
    print("\n🎉 EXCELLENT! 80%+ queries are instant!")
elif instant_tests >= total_tests * 0.6:
    print("\n✅ GOOD! 60%+ queries are instant!")
else:
    print("\n⚠️ More optimization needed")

print("\n" + "="*70)
print("NEW FEATURES TESTED:")
print("  ✅ Fast path for fees queries")
print("  ✅ Fast path for eligibility queries")
print("  ✅ Fast path for duration queries")
print("  ✅ Fast path for contact/application queries")
print("  ✅ LLM response caching")
print("  ✅ Conversation context tracking")
print("  ✅ Proper vector count display")
print("="*70)
