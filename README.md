## Connect 4

This is an implementation of the algorithm behind AlphaGo Zero (Nature volume 550, pages 354–359, doi:10.1038/nature24270) to learn "Connect 4" purely from self-play and without any prior knowledge of the game.

To be used as follows:
```
Options:
  -h, --help            show this help message and exit
  -t, --train           train a new model or resume training of an already
                        existing model [if specified with -f]
  -f MODEL_FILE, --file=MODEL_FILE
                        specifies a model to use for training, benchmarking or
                        manual play
  -c, --computer        launches a new game against a model [specified with
                        -f]
  -m, --manual          launches a new game against a human opponent
  -b, --benchmark       benchmarks a model [specified with -f] against a
                        random player
```

If CUDA is installed and Tensorflow set up to run on a GPU, but you still want to force it to run on the CPU instead, simply run the main executable as
```
CUDA_VISIBLE_DEVICES="" optirun python ./main.py
```
