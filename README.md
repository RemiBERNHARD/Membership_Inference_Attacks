# Experience with MIA (Membership Inference Attacks)

This repository contains python scripts used to perform some tests with MIAs and the reinjection mechanism inspired from cognitive science<sup>[1](#reinj)</sup> 

## Environment and libraries

The python scripts were executed in the following environment:

* OS: CentOS Linux 7
* GPU: NVIDIA GeForce GTX 1080 
* Cuda version: 9.0.176
* Python version: 2.7.5

The following version of some Python packages are necessary: 

* Tensorflow: 1.12.0
* Cleverhans: 3.0.1
* Keras: 2.2.4
* Numpy: 1.16.12


## File structure

This repository is divided at a high-level based on the data set considered (MNIST, Fashion-MNIST and CIFAR10). Within each folder associated to a data set, one must create a "models", "weights" and "indices" folder.

As an exemple for the MNIST data set:
  
1. To get target and shadow indices set of size 4,000 and 5,000 respectively:

        python get_indices.py 4000 5000
        
2. To train a target model:

        python train_mnist.py 0

(0 specifices the GPU to use)

3. To train the shadow model (to choose thresholds values when using MIAs against the target model):

        python train_mnist_shadow.py 0

3. To attack the target model with the adversary having the shadow model:

        python attack_mnist.py 0 target shadow

3. To train an auto-hetero associative model, and then train a model with noise injected through the auto-hetero associative model

        python train_mnist_auto_hetero.py 0 
        python train_mnist_reinj.py 0 
        
The files "attack_base.py", "attacks_threshold_choice.py" and "utils_mia.py" contain the implementation of various MIA methods.



<a name="reinj">1</a>: Miguel Solinas,  Stephane Rousset  , Romain Cohendet, Yannick Bourrier, Marion Mainsant, Anca Molnos, Marina Reyboz, and Martial Mermillod. Beneficial Effect of Combined Replay for Continual Learning. In *International Conference on Agents and Artificial Intelligence, 2021*
