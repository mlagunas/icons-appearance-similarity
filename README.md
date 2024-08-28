# learning-icons-appearance-similarity
Sample code that select similar icons using the method from the paper [Learning Icons Appearance Similarity](https://arxiv.org/abs/1902.05378).

#### Project structure

- `./small_dataset` folder containing a small subset downloaded from _the noun project_.
- `model_icons.pth` stores the weights of the model.
- `model_icons.py` contains the description of the model.
- `plot_similar.py` plots similar icons to a given reference.

#### How to run the code

First make sure that you have installed the following packages for python:

```
torch
torchvision
matplotlib
tqdm
```

**Download the weights of the trained model** using this [link](https://drive.google.com/file/d/1-zidFgj0kI7h3rK7uRDGsMSCglppJqX4/view?usp=sharing)
and make sure they are placed in the root folder of the project.
Then, in order to find similar icons in the given dataset run:
```
python3 plot_similar.py
```

If you did not modify the code, after running the script, a new folder _similar_icons_ will be created containing the images of the reference together with the _k_ closest icons to it in ascending order of distance (the distance is also written in the image title).

_Note that we have tested the code using Python 3.6_
#### Useful information

If you found this code useful please cite our work:
```
@Article{Lagunas2018,
  author="Lagunas, Manuel and Garces, Elena and Gutierrez, Diego",
  title="Learning icons appearance similarity",
  journal="Multimedia Tools and Applications",
  year="2018",
  month="Sep",
  issn="1573-7721",
  doi="10.1007/s11042-018-6628-7"
}
```

For more information and related work visit [my personal website](http://giga.cps.unizar.es/~mlagunas).

If you have further questions feel free to open an issue or send an email to `mlagunas at unizar dot es`.
