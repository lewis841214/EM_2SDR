
Environment :

Use protein_reconstruct as conda env


To run the EM-2SDR code for 3D reconstruction, one should first prepare:

1. N angles (Euler angle)

Then process these angles by Protein_reconstruction github's RotationMatrix

```

from cryoem.rotation_matrices import RotationMatrix

Orientation_Vectors = RotationMatrix(Angles)
#with open('Orientation_Vectors.pkl') as handle:
#    pickle.dump(Orientation_Vectors ,handle , protocol= pickle.HIGHEST_PROTOCOL )
```
one can refer to Projection.py in Protein_reconstruction github.

One should use Orientation_Vectors as the angle information.


2. Mean subtracted protein images:

One should preprocess the images.



