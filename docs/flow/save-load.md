# Save/load

The Flow contains a platform-agnostic computation Graph. It can be exported and imported between runtimes, languages, or even machines.
Here I'll go through the very simple interface Matcha provides for saving and loading the Flow. Let's first create a dummy Flow we'll work with:

```cpp
auto foo = (Flow) [](tensor x) {
  return x.t().dot(x);
};

tensor x = tensor::ones(5);
tensor y = foo(x);
```

I have also built the flow right away by calling it. You won't be able to save the flow without **having it built** first.
Now, to save it, simply give it the target file:

```cpp
foo.save("foo.matcha");
```

If the filename does not end with the `.matcha` extension, Matcha automatically appends it. The reverse process is equally simple.
To load a Flow back from a file, run:

```cpp
Flow bar = matcha::load("foo");
```

As with saving, the `.matcha` extension is appended to the filename if not present. From now on, we can use the loaded Flow
as if we have just created it:

```cpp
y = bar(x);

// bar.requireGrads(...);
// bar.save("more_bar");
```
