# CP and Tucker Decomposition for Pytorch CNN

## Objective 
The goal of this repository is to decompose each convolutional layer of convolutional neural networks and thus reduce computational time. More specifically, we want to see the reduction of number of floating point calculation (Flops), number of multiply-adds, inference evaluation time and CPU prediction time. 

We focus on 2 popular tensor decomposition techniques: CP decomposition and Tucker decomposition. Moreover, we aim to compare the performance of CNN with and without decompositions. This is a final project for the course Numerical Linear Algebra, Skoltech, Fall 2022. 

## CNN Architecture 
- Resnet18
- Densenet

# Requirements
Below are the list of required packages to run this repository:
- torch 1.13.1
- torchstat 0.0.7
- tensorly 0.7.0

## Usage
There are several command line arguments. 
```bash
python3 main.py [-mode MODE] [-decompose_mode DECOMPMODE] [-cnn_type CNNMODEL] [-num_epoches EPOCH] [-lr LR]
```
- **MODE** specifies the code to decompose, train, or evaluate (normal evaluate without decomposition or with decomposition). The options are below:
  - `decompose`: decompose the CNN with either CP or Tucker decompositon. If this mode is used, **DECOMPMODE** is required (with default being Tucker decompsition)
  - `train`: train the CNN without decomposition
  - `evaluate_none`: evaluate the CNN without decomposition
  - `evaluate_decomposed`: evaluate the CNN with either Tucker or CP decomposition (once again, **DECOMPMODE** is required). If this mode is used, it is assume that 'decompose' has already ran and the compressed network weights are already saved in `checkpoints`.
- **DECOMPMODE** specifies the decomposition (either Tucker or CP, with default being Tucker)
- **CNNMODEL** specifies the CNN architecture (either Resnet18 or Densenet, with default being Resnet18)
- **EPOCH** specifies the number of epoch to train 
- **LR** specifies the learning rate of the optimizer

## Results
#### Computational Result
| Model | Decomposition | Flops | mAdds | Inf time | CPU Pred Time
| :----------- | :-----------: | -----------: | :-----------: | :-----------: | :-----------: |
| Resnet18  |None   | 556.65MFlops | 1.11GMAdd    | 0:00:05.78605   | 0.0564924  |
| Resnet18  |Tucker | 298.7MFlops  | 596.39MMAdd  | 0:00:03.73833   | 0.0320413  |
| Resnet18  |CP     | 275.2MFlops  | 549.37MMAdd  | 0:00:03.565069  | 0.0304961  |
| Densenet  |None   | 128.98MFlops | 257.19MMAdd  | 0:00:08.402589  | 0.0756694  |
| Densenet  |Tucker | 75.18MFlops  | 149.39MMAdd  | 0:00:05.055992  | 0.0562944  |
| Densenet  |CP     | 73.31MFlops  | 145.67MMAdd  | 0:00:05.532252  | 0.0583822  | 

#### Accuracy results
The goal of this project is not to attain the highest accuracy. We solely wanted to see if we can reduce computational speed without affecting the performance. 
| Model | Decomposition | Accuracy
| :----------- | :-----------: | -----------: |
| Resnet18  |None   |  79.27%| 
| Resnet18  |Tucker |  80.21%| 
| Resnet18  |CP     |  79.21%|
| Densenet  |None   | 58.48%| 
| Densenet  |Tucker | 59.01% | 
| Densenet  |CP     |  58.58% | 
## References

https://arxiv.org/pdf/1905.10145.pdf 

https://arxiv.org/pdf/1701.07148.pdf

https://arxiv.org/pdf/1412.6553v3.pdf

https://arxiv.org/abs/1412.6553 

https://arxiv.org/abs/1511.06530

https://iksinc.online/tag/tucker-decomposition/

https://iksinc.online/tag/cp-decomposition/
