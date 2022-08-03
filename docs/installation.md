# Installation

Currently, you have to compile Matcha locally. Precompiled libraries with
aggressive optimizations for common architectures will be available in the future.

## Installation using CMake

The Matcha engine is available as a [GitHub repo](https://github.com/matcha-ai/matcha/). 
Its compressed `.zip` version can be downloaded [here](https://github.com/matcha-ai/matcha-engine/archive/refs/heads/main.zip).

First, download the repository or clone it, and go there:

```sh
git clone https://github.com/matcha-ai/matcha
cd matcha
```

Now install the project by executing the following command. Note that
it may require root privileges:

```sh
make install
```

This will configure the project with CMake, compile it, and install it.
