"""
Quick test to check Qdrant data structure
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

print("="*60)
print("QDRANT STRUCTURE TEST")
print("="*60)

# Get a few points to see the structure
print("\n1. Getting first 5 points (no filter)...")
results = client.scroll(
    collection_name="university_kb",
    limit=5,
    with_payload=True,
    with_vectors=False
)

print(f"\nFound {len(results[0])} points")

for i, point in enumerate(results[0][:3], 1):
    print(f"\n--- Point {i} ---")
    print(f"ID: {point.id}")
    print(f"Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
    if point.payload:
        print(f"Payload: {point.payload}")

# Try filtering by type
print("\n" + "="*60)
print("2. Testing filter by type='program'...")
print("="*60)

try:
    results = client.scroll(
        collection_name="university_kb",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="program")
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    print(f"\n✓ Found {len(results[0])} program points")
    if results[0]:
        print(f"\nFirst program payload:")
        print(results[0][0].payload)
except Exception as e:
    print(f"\n❌ Error with type filter: {e}")
    import traceback
    traceback.print_exc()

# Try filtering by metadata.type
print("\n" + "="*60)
print("3. Testing filter by metadata.type='program'...")
print("="*60)

try:
    results = client.scroll(
        collection_name="university_kb",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value="program")
                )
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    print(f"\n✓ Found {len(results[0])} program points")
    if results[0]:
        print(f"\nFirst program payload:")
        print(results[0][0].payload)
except Exception as e:
    print(f"\n❌ Error with metadata.type filter: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DONE")
print("="*60)
