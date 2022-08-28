# Flatten layer
> `nn::flatten(const tensor&) -> tensor` \
> `struct nn::Flatten`

Flattens the input to the shape `{bsize, -1}`, where `bsize` is
the input batch size. 

?> Use the `flatten` before feeding the data
   to fully connected layers. This will make it possible for
   such network architectures to process
   data of dimensionality greater than 1 - for example, to process
   2D images or 3D space maps.
