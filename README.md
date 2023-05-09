# ORIE 6125 Final Project

This is a solver for financial stocks information. Particularly, we consider three RL methods: DQN, Double DQN and Duel Double DQN.

Notice that the main difference between theses three methods is how we update the Q value.

Here we give an illustration of the main file result.ipynb.

1. First given a stock file name, we can visualize the change of stock price over a period, and we equally separate the date into two parts, where the first part is training set, and the second part aims for testing.
! [Screenshot of a stock prices, green for price increases and red for price decreases.](http)

