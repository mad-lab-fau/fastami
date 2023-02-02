[![codecov](https://codecov.io/gh/mad-lab-fau/fastami/branch/main/graph/badge.svg?token=U379I88TBU)](https://codecov.io/gh/mad-lab-fau/fastami)
# FastAMI

A Monte Carlo approximation to the adjusted and standardized mutual information for faster clustering comparisons. You can use this package as a drop-in replacement for ``skleran.metrics.adjusted_mutual_info_score``, when the exact calculation is too slow, i.e. because of large datasets and large numbers of clusters.

## Installation

``fastami`` requires Python >=3.8. You can install ``fastami`` via pip from PyPI:

```bash
pip install fastami
```

## Usage Examples

### FastAMI

You can use FastAMI as you would use ``adjusted_mutual_info_score`` from ``scikit-learn``:

```python
from fastami import adjusted_mutual_info_mc

labels_true = [0, 0, 1, 1, 2]
labels_pred = [0, 1, 1, 2, 2]

ami, ami_error = adjusted_mutual_info_mc(labels_true, labels_pred)

# Output: AMI = -0.255 +- 0.008
print(f"AMI = {ami:.3f} +- {ami_error:.3f}")
```

Note that the output may vary a little bit, due to the nature of the Monte Carlo approach. If you would like to ensure reproducible results, use the ``seed`` argument. By default, the algorithm terminates when it reaches an accuracy of ``0.01``. You can adjust this behavior using the ``accuracy_goal`` argument.

### FastSMI

FastSMI works similarly:

```python
from fastami import standardized_mutual_info_mc

labels_true = [0, 0, 1, 1, 2]
labels_pred = [0, 1, 1, 2, 2]

smi, smi_error = standardized_mutual_info_mc(labels_true, labels_pred)

# Output: SMI = -0.673 +- 0.035
print(f"SMI = {smi:.3f} +- {smi_error:.3f}")
```

While FastSMI is usually faster than an exact calculation of the SMI, it is still orders of magnitude slower than FastAMI. Since the SMI is not confined to the interval ``[-1,1]`` like the AMI, the SMI by default terminates at a given absolute or relative error of at least ``0.1``, whichever is reached first. You can adjust this behavior using the ``precision_goal`` argument.

## Citing FastAMI

If you use `fastami` in your research work, please cite the corresponding paper (will probably be published by March 2023):

```
Klede et al., (2023). FastAMI - A Monte Carlo Approach to the Adjustment for Chance in Clustering Comparison Metrics. Proceedings of the AAAI Conference on Artificial Intelligence.
```
