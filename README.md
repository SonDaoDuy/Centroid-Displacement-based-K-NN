# Centroid-Displacement-based-K-NN
This is a python implementation of Centroid Displacement-based k-Nearest Neigbors Algorithm (CDNN) which is depend on the idea of the paper [Robust Biometric Recognition From Palm Depth Images for Gloved Hands](https://ieeexplore.ieee.org/document/7161357).

The repositry includes:
- CDNN implementation by using numpy
- Examples of using CDNN
- A comparision between CDNN and tradditional k-NN algorithm on some sample datasets

To run the test, install all modules in `requirement.txt` and run:
```
python test.py
```

A sample result will look like this:
```
Testing with k = 20

---------------Digits dataset------------------
Loading data.....
Done loading data!

Number of classes: 10
Data dimension: 64
Number of training samples: 1437
Number of testing samples: 360

Predict time for CDNN: 3.553s
Accuracy for CDNN with k = 20: 0.967

Predict time for kNN with uniform weights: 0.085s
Accuracy for kNN with k = 20 and uniform weights: 0.964

Predict time for kNN with distance weights: 0.062s
Accuracy for kNN with k = 20 and distance weights: 0.964

-----------------------------------------------
```

## Citation
If you use this code or CDNN algorithm for your research, please cite this paper.
```
@article{nguyen2015robust,
  title={Robust biometric recognition from palm depth images for gloved hands},
  author={Nguyen, Binh P and Tay, Wei-Liang and Chui, Chee-Kong},
  journal={IEEE Transactions on Human-Machine Systems},
  volume={45},
  number={6},
  pages={799--804},
  year={2015},
  publisher={IEEE}
}
```

