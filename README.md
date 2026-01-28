# micro-tools
Small tools I use for specialized tasks.

## critbinom.py

A precise replacement for the CRITBINOM() function in Excel/Google Sheets, designed for very low criterion values (alpha). I created this to calculate values for the NIST APT test for entropy sources where alpha is below $2^{-53}$, a value that cannot be represented by the standard `double` type.

## asm_cleaner.py

I created this while optimizing computation performance. I found that compiler listings (e.g., `clang -S` or `gcc -S`) were noisy and difficult to read, while `llvm-objdump` failed to display symbolic names properly despite providing convenient inline C/C++ code. This tool attempts to combine the best of both worlds by converting `clang -S` output into clean assembly code, complete with references to the original C/C++ code and useful compiler comments regarding register states.

## sortlib.py

I maintain a local collection of interesting papers and books. Eventually, the unsorted collection grew to thousands of documents, many of which I wasn't actively using, making sorting a burden. This tool was born to solve that. Developing a taxonomy for my library was a fruitful exercise that forced me to clarify my actual goals. Once the structure was in place, sorting became easy, allowing me to upload specific thematic folders into tools like NotebookLM.

## concat.py

A tool to combine source code into a single large text file, with each file clearly marked for LLM consumption. This enables the use of code folders in tools like NotebookLM, which currently cannot process raw code repositories directly.

## objsize.py

Post-processes `objdump` output to calculate the size of `.text` and `.rodata` sections. This is useful for calculating total flash memory usage.

## symsize.py

A more advanced tool for breaking down code size contributed by libraries. It takes a list of root functions (like `main`) and recursively looks for dependencies, locating them in object files and sections to calculate the total byte cost of using a specific function.

