"""Check what programs exist"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from AIVoiceAssistant_new import AIVoiceAssistant

a = AIVoiceAssistant('university_dataset_advanced.json')
progs = a._get_all_from_qdrant('program')

print("Programs with 'AI' or 'Artificial':")
for p in progs:
    if 'ai' in p.lower() or 'artificial' in p.lower():
        print(f"  - {p}")

print("\nPrograms with 'B.Tech':")
for p in progs:
    if 'b.tech' in p.lower():
        print(f"  - {p}")
