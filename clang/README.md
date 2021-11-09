# clang

## specify libclang dll path
[Why can't this python script find the libclang dll?](https://stackoverflow.com/questions/22730935/why-cant-this-python-script-find-the-libclang-dll)
```python
import clang.cindex

clang.cindex.Config.set_library_file('D:/LLVM/bin/libclang.dll')
```
