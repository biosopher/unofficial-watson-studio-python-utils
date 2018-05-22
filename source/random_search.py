# Random search is surprisingly effective at finding the optimal hyperparameters.  See this blog for more details:
# http://www.cnblogs.com/yymn/p/4536740.html
#
#
# Quote from the blog:
# "I love movies where the underdog wins, and I love machine learning papers where simple solutions are shown to be
# surprisingly effective. This is the storyline of “Random search for hyperparameter optimization”
# by Bergstra and Bengio. Random search is a slight variation on grid search. Instead of searching over the entire
# grid, random search only evaluates a random sample of points on the grid. This makes random search a lot cheaper
# than grid search. Random search wasn’t taken very seriously before. This is because it doesn’t search over all the
# grid points, so it cannot possibly beat the optimum found by grid search. But then came along Bergstra and Bengio.
# They showed that, in surprisingly many instances, random search performs about as well as grid search. All in all,
# trying 60 random points sampled from the grid seems to be good enough.

# In hindsight, there is a simple probabilistic explanation for the result: for any distribution over a sample space
# with a finite maximum, the maximum of 60 random observations lies within the top 5% of the true maximum,
# with 95% probability. That may sound complicated, but it’s not. Imagine the 5% interval around the true maximum.
# Now imagine that we sample points from this space and see if any of it lands within that maximum. Each random draw
# has a 5% chance of landing in that interval, if we draw n points independently, then the probability that
# all of them miss the desired interval is (1−0.05)n. So the probability that at least one of them succeeds in hitting
# the interval is 1 minus that quantity. We want at least a .95 probability of success. To figure out the
# number of draws we need, just solve for n in the equation"

import random

class RandomSearch:

    def __init__(self):

        self.params_ranges = {}
        self.statics = {}
        self.params_powers = {}
        self.hpo_lists = {}

    def add_step_range(self, name, min_val, max_val, step):
        self.params_ranges[name] = [min_val, max_val, step]

    def add_power_range(self, name, min_val, max_val, power):
        self.params_powers[name] = [min_val, max_val, power]

    def add_static_var(self, name, value):
        self.statics[name] = value

    def add_list(self, name, value_list):
        self.hpo_lists[name] = value_list

    def create_random_search(self, search_count):

        random_search= []
        for i in range(0, search_count):
            hyperparameters = {}
            random_search.append(hyperparameters)

            # add value for ranges
            for name in self.params_ranges:
                min_val = self.params_ranges[name][0]
                max_val = self.params_ranges[name][1]
                step = self.params_ranges[name][2]
                values = []
                value = min_val
                while value <= max_val:
                    values.append(value)
                    value += step
                index = random.randint(0, len(values)-1)
                hyperparameters[name] = values[index]

            # add value for powers
            for name in self.params_powers:
                min_val = self.params_powers[name][0]
                max_val = self.params_powers[name][1]
                power = self.params_powers[name][2]
                powers = []
                for index in range(min_val, max_val + 1):
                    powers.append(power ** index)
                index = random.randint(0, len(powers)-1)
                hyperparameters[name] = powers[index]

            # add lists
            for name in self.hpo_lists:
                value_list = self.hpo_lists[name]
                index = random.randint(0, len(value_list)-1)
                hyperparameters[name] = value_list[index]

            # add static variables
            for name in self.statics:
                hyperparameters[name] = self.statics[name]

        return random_search
