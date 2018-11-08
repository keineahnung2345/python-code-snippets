# python-code-snippets
Some useful python code snippets

## to remove non-ascii characters from a string
```python
s = s.encode('ascii', errors='ignore').decode()
```
