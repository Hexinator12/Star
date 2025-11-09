# Performance Optimization Guide

## Recent Changes (Nov 9, 2025)

### Timeout & Context Improvements

To handle the larger knowledge base (`university_dataset_advanced.json` with 1,252 vectors), we've made the following optimizations:

#### 1. **Increased Timeout**
```python
# AIVoiceAssistant_new.py
request_timeout=120.0  # Increased from 30s to 120s
```

#### 2. **Increased Context Window**
```python
num_ctx=4096  # Increased from 2048 to 4096
```

#### 3. **Reduced Retrieved Documents**
```python
similarity_top_k=5  # Reduced from 10 to 5
```
#### 3. **Reduced Retrieved Documents**
```python
model="phi"  # Change from gemma:2b to change model
```
**Larger Knowledge Base = More Processing Time**
- **1,252 vectors** vs previous ~157 vectors
- More documents to search through
- Richer content per document
- More context to process

**Trade-offs:**
- ✅ **Better:** More time for complex queries
- ✅ **Better:** Larger context window for detailed responses
- ⚠️ **Trade-off:** Slightly slower responses (but more accurate)
- ✅ **Better:** Fewer documents = faster LLM processing

## Performance Tips

### 1. **Ask Specific Questions**

❌ **Avoid broad questions:**
- "Tell me everything about the university"
- "What are all the programs?"

✅ **Ask specific questions:**
- "What programs are available in engineering?"
- "Tell me about B.Tech AI"
- "What are the fees for MBA?"

### 2. **Ollama Model Selection**

Current model: `gemma:2b` (fast, lightweight)

**Alternative models:**

```bash
# Faster (less accurate)
ollama pull phi

# Balanced (current)
ollama pull gemma:2b

# More accurate (slower)
ollama pull llama2:7b
ollama pull mistral
```

To change model, edit `AIVoiceAssistant_new.py`:
```python
self.llm = Ollama(
    model="phi",  # Change this
    ...
)
```

### 3. **Adjust Retrieved Documents**

In `AIVoiceAssistant_new.py`, adjust `similarity_top_k`:

```python
# Faster (less context)
similarity_top_k=3

# Balanced (current)
similarity_top_k=5

# More context (slower)
similarity_top_k=10
```

### 4. **Response Mode Options**

Current: `response_mode="compact"`

**Options:**
- `"compact"` - Fast, concise (current)
- `"refine"` - Slower, more detailed
- `"tree_summarize"` - Best for long documents

### 5. **Reduce Context Window**

If responses are too slow:

```python
num_ctx=2048  # Reduce from 4096
```

## Troubleshooting

### Issue: "ReadTimeout" Error

**Cause:** Query taking too long to process

**Solutions:**
1. ✅ **Already done:** Increased timeout to 120s
2. Ask more specific questions
3. Reduce `similarity_top_k` to 3
4. Use faster model (phi)
5. Reduce `num_ctx` to 2048

### Issue: Out of Memory

**Cause:** Too much context loaded

**Solutions:**
1. Reduce `num_ctx` to 2048 or 1024
2. Reduce `similarity_top_k` to 3
3. Use smaller model (phi or gemma:2b)

### Issue: Slow Responses

**Cause:** Large knowledge base + detailed processing

**Solutions:**
1. Use faster model: `phi`
2. Reduce `similarity_top_k` to 3
3. Use `response_mode="compact"`
4. Ask more specific questions

## Monitoring Performance

### Check Response Times

The system logs include timing information:

```
Query: "What programs are available?"
Retrieved: 5 documents
Processing time: ~5-10 seconds (normal)
```

### Expected Response Times

With current settings:

| Query Type | Expected Time |
|------------|---------------|
| Simple (cached) | < 1 second |
| Simple (new) | 3-5 seconds |
| Complex | 5-15 seconds |
| Very complex | 15-30 seconds |

### If Timeout Still Occurs

1. **Increase timeout further:**
   ```python
   request_timeout=180.0  # 3 minutes
   ```

2. **Use streaming responses** (future enhancement):
   ```python
   stream=True
   ```

3. **Implement query preprocessing** to break down complex queries

## Hardware Recommendations

### Minimum:
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Storage:** 10 GB free

### Recommended:
- **CPU:** 8+ cores
- **RAM:** 16 GB
- **Storage:** 20 GB free
- **GPU:** Optional (for faster embeddings)

## Future Optimizations

### Planned Improvements:

1. **Streaming Responses**
   - Show partial results as they're generated
   - Better user experience

2. **Query Caching**
   - Cache common queries
   - Faster repeat questions

3. **Async Processing**
   - Non-blocking queries
   - Better concurrency

4. **Smart Document Filtering**
   - Pre-filter by metadata
   - Reduce search space

5. **Hybrid Search**
   - Combine semantic + keyword search
   - Better accuracy + speed

## Configuration Summary

### Current Optimized Settings:

```python
# LLM Configuration
model = "gemma:2b"
request_timeout = 120.0
temperature = 0.1
num_ctx = 4096

# Retrieval Configuration
similarity_top_k = 5
response_mode = "compact"

# Knowledge Base
file = "university_dataset_advanced.json"
vectors = 1252
embedding_dim = 384
```

### For Maximum Speed:

```python
model = "phi"
request_timeout = 60.0
num_ctx = 2048
similarity_top_k = 3
```

### For Maximum Accuracy:

```python
model = "llama2:7b"
request_timeout = 180.0
num_ctx = 4096
similarity_top_k = 10
response_mode = "refine"
```

## Testing Performance

### Quick Test Script:

```python
import time
from AIVoiceAssistant_new import AIVoiceAssistant

assistant = AIVoiceAssistant()

queries = [
    "What programs are available?",
    "Tell me about B.Tech AI",
    "What are the fees?",
]

for query in queries:
    start = time.time()
    response = assistant.interact_with_llm(query)
    elapsed = time.time() - start
    print(f"Query: {query}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Response length: {len(response)} chars\n")
```

## Contact & Support

If you continue experiencing timeout issues:

1. Check Ollama is running: `ollama list`
2. Check Qdrant is running: Visit `http://localhost:6333/dashboard`
3. Review logs for specific errors
4. Try the "Maximum Speed" configuration above

---

**Last Updated:** November 9, 2025  
**Version:** 2.0 (Advanced Dataset)
