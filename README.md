# Neural network with genetic algorithm

Implementation of neural network with genetic algorithm.


The program takes 8 parametars and they all have to be specified when running the program:
<ol>
  <li> --train: file to train the NN </li>
  <li> --test: file to test the NN </li>
  <li> --n: layers of neural network split with 's' eg. NN with 2 hidden layer 4 and 5 would be '4s5s' </li>
  <li> --popsize: size of population </li>
  <li> --elitism: how many parents we want to add to next generation </li>
  <li> --p: probability of mutation for elements of a chromosome </li>
  <li> --K: standard deviation in mutation fuction </li>
  <li> --iters: number of iterations to run </li>
</ol>
Example:

```console
foo@bar:~$ python solution.py --train ./files/sine_train.txt --test ./files/sine_test.txt --nn 5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000
```

It also writes out error of the best individual in a population every 2000 iterations.

The only package that is needed to run the program is [`numpy`](https://numpy.org/).
