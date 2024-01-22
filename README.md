# Machine Learning for Quantitative Finance

Source codes for my submitted project report: *Machine Learning for Quantitative Finance*. 

The codes contain:
1. Replications of methodology and results in Lu(23)'s paper "*Temporal difference learning for high-dimensional PIDES with jumps*"

2. Applications of Lu(23)'s methodology on the pricing of various options, namely,

   - Asian options of basket of n stocks,

   - Barrier options, 

   - up-and-out put options (in Hong Kong, it is also called callable bull bear contracts), and 

   - American up-and-out put options.

## Requirements

Before running the codes, you need to create a new conda environment first, with the following packages:

* python == 3.8
* Tensorflow == 2.4
* Cudnn == 8.0
* cudatoolkit == 11.0
* pandas
* numpy == 1.23.5, or 1.19.5 (to run for American up-and-out put options)
* scipy
* wandb (optional, to track and visualize your training process)
* munch
* tqdm

## How to test
Please go to: 

[Replication](./Lu/README.md), 

[Asian options](./Asian%20options/README.md),

[Barrier options](./Barrier%20options/README.md), 

for detailed instructions.

## Experiments

Codes for the **replications of Lu(23)**'s methodology and results are in `Lu/`. It contains:

* One dimensional pure jump process;
* Robustness check under different parameter settings, e.g. different numbers of trajectories M and time intervals N;
* High dimensional (100-dimensional) problems.

<br/>

Codes for applying Lu(23)'s method on **Asian options** are in `Asian options/`.

<br/>

Codes for applying Lu(23)'s method on various **Barrier options** can be found in `Barrier options/`, which are 
* a down-and-in option
* a down-and-out option
* an up-and-in option
* an up-and-out option
* an American up-and-out put option

## Reference
[1] Lu, L., Guo, H., Yang, X., & Zhu, Y. (2023). Temporal difference learning for high-dimensional PIDEs with jumps. arXiv preprint arXiv:2307.02766.

[2] Gnoatto, A., Patacca, M., & Picarelli, A. (2022). A deep solver for BSDEs with jumps. arXiv preprint arXiv:2211.04349.

## Contact
If you have any questions regarding the code, please create an issue or email me at yqdeng@connect.hku.hk.
