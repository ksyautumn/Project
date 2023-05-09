class TradingEnv:
    def __init__(self, data, window_size=90):
        self.data = data
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.pos = []
        self.position_value = 0
        self.history = [0 for _ in range(self.window_size)]
        return [self.position_value] + self.history
    
    def step(self, action):
        reward = 0
        
        # We define action = 0: stay, 1: buy, 2:sell
        if action == 1:
            self.pos.append(self.data.iloc[self.t,:]['Close'])
        elif action == 2:
            if len(self.pos) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.pos:
                    profits += (self.data.iloc[self.t,:]['Close']-p)
                reward += profits
                self.profits += profits
                self.pos = []
                
        # Set next time
        self.t += 1
        self.position_value = 0
        for p in self.pos:
            self.position_value += (self.data.iloc[self.t,:]['Close']-p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t,:]['Close']-self.data.iloc[(self.t-1),:]['Close'])
        
        # We define reward to be +1 if gain money, -1 if lose money
        
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
            
        return [self.position_value] + self.history,reward,self.done