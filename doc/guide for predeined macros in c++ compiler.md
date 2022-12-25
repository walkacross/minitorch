# Guide to predefined macros in C++ compilers (gcc, clang, msvc etc.)
> https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

When writing portable C++ code you need to write conditional code that depends on compiler used or the OS for which the code is written.
Here’s a typical case:
~~~
#if defined (_MSC_VER)
// code specific to Visual Studio compiler
#endif
~~~
To perform those checks you need to check pre-processor macros that various compilers set.
It can either be binary is defined vs. is not defined check (e.g. __APPLE__) or checking a value of the macro (e.g. _MSC_VER defines version of Visual Studio compiler).
This document describes macros set by various compilers.

## Checking for OS (platform)
To check for which OS the code is compiled:
~~~
Linux and Linux-derived           __linux__
Android                           __ANDROID__ (implies __linux__)
Linux (non-Android)               __linux__ && !__ANDROID__
Darwin (Mac OS X and iOS)         __APPLE__
Akaros (http://akaros.org)        __ros__
Windows                           _WIN32
Windows 64 bit                    _WIN64 (implies _WIN32)
NaCL                              __native_client__
AsmJS                             __asmjs__
Fuschia                           __Fuchsia__
~~~

## Checking the compiler type:
To check which compiler is used:
~~~
Visual Studio       _MSC_VER
gcc                 __GNUC__
clang               __clang__
emscripten          __EMSCRIPTEN__ (for asm.js and webassembly)
MinGW 32            __MINGW32__
MinGW-w64 32bit     __MINGW32__
MinGW-w64 64bit     __MINGW64__
~~~

## Checking compiler version

### gcc
__GNUC__ (e.g. 5) and __GNUC_MINOR__ (e.g. 1).
To check that this is gcc compiler version 5.1 or greater:
~~~
#if defined(__GNUC__) && (__GNUC___ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ >= 1))
// this is gcc 5.1 or greater
#endif
~~~
Notice the chack has to be: major > 5 || (major == 5 && minor >= 1). If you only do major == 5 && minor >= 1, it won’t work for version 6.0.

### clang
__clang_major__, __clang_minor__, __clang_patchlevel__

### Visual Studio
_MSC_VER and _MSC_FULL_VER: