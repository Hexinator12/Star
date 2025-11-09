"""
Quick Speed Test Script
Tests the performance of different query types
"""

import time
from AIVoiceAssistant_new import AIVoiceAssistant

def test_query(assistant, query, expected_time, description):
    """Test a single query and measure time."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Query: '{query}'")
    print(f"Expected: < {expected_time}s")
    print(f"{'='*60}")
    
    start = time.time()
    response = assistant.interact_with_llm(query)
    elapsed = time.time() - start
    
    # Truncate long responses
    response_preview = response[:200] + "..." if len(response) > 200 else response
    
    print(f"\n✓ Response ({elapsed:.2f}s):")
    print(response_preview)
    
    if elapsed < expected_time:
        print(f"\n✅ PASS: {elapsed:.2f}s < {expected_time}s")
    else:
        print(f"\n⚠️ SLOW: {elapsed:.2f}s >= {expected_time}s")
    
    return elapsed

def main():
    print("\n" + "="*60)
    print("RAG AI ASSISTANT - SPEED TEST")
    print("="*60)
    
    print("\n🚀 Initializing assistant...")
    start_init = time.time()
    assistant = AIVoiceAssistant("university_dataset_advanced.json")
    init_time = time.time() - start_init
    print(f"✓ Initialized in {init_time:.2f}s")
    
    results = []
    
    # Test 1: List all programs (should be INSTANT)
    results.append(test_query(
        assistant,
        "List all programs",
        0.5,
        "INSTANT - List all programs (Direct Qdrant)"
    ))
    
    # Test 2: Count programs (should be INSTANT)
    results.append(test_query(
        assistant,
        "How many programs?",
        0.5,
        "INSTANT - Count programs (Direct Qdrant)"
    ))
    
    # Test 3: What programs available (should be INSTANT)
    results.append(test_query(
        assistant,
        "What programs are available?",
        0.5,
        "INSTANT - What programs available (Direct Qdrant)"
    ))
    
    # Test 4: Specific program (should be FAST)
    results.append(test_query(
        assistant,
        "Tell me about B.Tech AI",
        5.0,
        "FAST - Specific program info (RAG + LLM)"
    ))
    
    # Test 5: Fees query (should be FAST)
    results.append(test_query(
        assistant,
        "What are the fees for MBA?",
        5.0,
        "FAST - Fees query (RAG + LLM)"
    ))
    
    # Test 6: Cached query (should be INSTANT)
    results.append(test_query(
        assistant,
        "List all programs",
        0.2,
        "INSTANT - Cached query (from cache)"
    ))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_time = sum(results) / len(results)
    instant_queries = sum(1 for t in results[:3] if t < 0.5)
    fast_queries = sum(1 for t in results[3:5] if t < 5.0)
    
    print(f"\nTotal queries: {len(results)}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Instant queries (< 0.5s): {instant_queries}/3")
    print(f"Fast queries (< 5s): {fast_queries}/2")
    print(f"Cached queries (< 0.2s): {1 if results[5] < 0.2 else 0}/1")
    
    if instant_queries == 3 and fast_queries == 2:
        print("\n✅ ALL TESTS PASSED! System is production-ready! 🚀")
    else:
        print("\n⚠️ Some tests were slower than expected. Check configuration.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
