# ORIE 6125 Final Project

This is a solver for financial stocks information. Particularly, we consider three RL methods: DQN, Double DQN and Duel Double DQN.

Notice that the main difference between theses three methods is how we update the Q value. 

Here we give an illustration of the main file result.ipynb.

1. First given a stock file name, we equally separate the date into two parts, with the first part begin training set, and the second part aims for testing (with commend  `[train, test, date_split] = read_file(filename)`), we can also visualize the change of stock price over a period with commend `plot_train_test(train, test, date_split)`
! [Screenshot of a stock prices, green for price increases and red for price decreases.](http)

2. We wrap up one RL model for one stock in the report function, for example by coding `DQN_report(train)` we can give a list result including the inter-stage report of profit, execute time for the programming, and also a record for the loss and revenue at difference epoches. We also visualize the training result. Particularly, in our environment we have three possible movements in each step: we can either buy (in gray), sell (in cyan) or hold (magenta).

3. Similarly we can code `DoubleDQN_report(train)` and `DuelDouDQN_report(train)`


