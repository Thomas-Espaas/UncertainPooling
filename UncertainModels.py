import casadi
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.special import erf as scp_erf
from scipy.stats import norm
import time
import pandas as pd


def norm_cdf(x, mu=0, std=1):
    """
    Function to return the value of the cumulative distribution function for a normally distributed random variable
    :param x: The value at which we evaluate the CDF
    :param mu: The mean of the distribution. Defaults to to standard normal distribution
    :param std: The standard deviation of the distribution. Defaults to to standard normal distribution
    :return: Float. The value of the cumulative distribution function given the above parameters
    """

    f = (1/2) * (1 + scp_erf((x - mu)/(std*np.sqrt(2))))
    return f


class UncertainModel:
    """
    The parent class for the various uncertain models. Includes amongst other things a method for evaluating the the
    true stochastic solution given a set of variable values. Usually these variable values will be given by optimising
    the proxy model. It also includes a method for saving the solution statistics to a results file
    """

    def __init__(self, problem_name, std=0.02):
        self.problem_name = problem_name
        self.std = std
        self.problem_status = 'unsolved'

    def solve_problem(self):
        """
        Since the structure of the problem solving method depends quite strongly on the model used, this is simply a
        placeholder, meant to be overridden in the child classes.
        """
        pass

    def ss_evaluator(self):
        """
        Method to calculate the objective metric of the stochastic model respecting the true nature of uncertainty.
        It uses the value for the variables determined through the optimization of a proxy model.
        """
        if self.problem_status == 'unsolved':
            print('Problem has not been solved using an optimisation under uncertainty method yet.')
            print('Optimise the problem using one of the available methods first.')

            exit()

        if self.problem_name == 'Haverly_1':
            feed_price = [6, 16, 10]
            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            # Load problem stats and data that will already have been set or obtained through optimisation

            feed_mus = self.feed_mus
            feed_stds = [self.std, self.std, self.std]

            proxy_solution = self.proxy_solution['x']

            feed_flows = [proxy_solution['feed_flows[0]'], proxy_solution['feed_flows[1]'],
                          proxy_solution['feed_flows[2]']]

            pool_flows = [proxy_solution['pool_flows[0]'], proxy_solution['pool_flows[1]'],
                          proxy_solution['pool_flows[2]'], proxy_solution['pool_flows[3]']
                          ]
            product_flows = [proxy_solution['product_flows[0]'], proxy_solution['product_flows[1]']]

            # The below calculates the compositions and standard deviations of compositions throughout the Haverly
            # network. Some handling must be done in case flows into a pool or product is zero (to avoid division
            # by zero errors)

            pool_mus = [0, 0]
            pool_stds = [0, 0]

            try:
                pool_mus[0] = (feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1]) / (
                            feed_flows[0] + feed_flows[1])
                pool_stds[0] = np.sqrt(
                    (feed_flows[0] ** 2 * feed_stds[0] ** 2 + feed_flows[1] ** 2 * feed_stds[1] ** 2) / (
                                feed_flows[0] + feed_flows[1]) ** 2)
            except:
                pool_mus[0] = 0
                pool_stds[0] = 1

            try:
                pool_mus[1] = feed_mus[2]
                pool_stds[1] = feed_stds[2]
            except:
                pool_mus[1] = 0
                pool_stds[1] = 1

            product_mus = [0, 0]
            product_stds = [0, 0]

            try:
                product_mus[0] = sum(pool_flows[i * 2 + 0] * pool_mus[i] for i in range(2)) / product_flows[0]
                product_stds[0] = np.sqrt(
                    sum(pool_flows[i * 2 + 0] ** 2 * pool_stds[i] ** 2 for i in range(2)) / product_flows[0] ** 2)
            except:
                product_mus[0] = 0
                product_stds[0] = 1

            try:
                product_mus[1] = sum(pool_flows[i * 2 + 1] * pool_mus[i] for i in range(2)) / product_flows[1]
                product_stds[1] = np.sqrt(
                    sum(pool_flows[i * 2 + 1] ** 2 * pool_stds[i] ** 2 for i in range(2)) / product_flows[1] ** 2)
            except:
                product_mus[1] = 0
                product_stds[1] = 1

            # Probabilities of the two products satisfying their quality constraints can be calculated using the
            # cumulative distribution function

            probabilities = [0, 0]

            probabilities[0] = norm_cdf(self.product_qualities[0], product_mus[0], product_stds[0])
            probabilities[1] = norm_cdf(self.product_qualities[1], product_mus[1], product_stds[1])

            self.stochastic_solution = (sum(feed_flows[i] * feed_price[i] for i in range(3)) -
                                        sum(probabilities[i] * product_flows[i] * product_price[i] for i in range(2)))
        elif self.problem_name == 'Haverly_2':
            feed_price = [6, 16, 10]
            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            # Load problem stats and data that will already have been set or obtained through optimisation

            feed_mus = self.feed_mus
            feed_stds = [self.std, self.std, self.std]

            proxy_solution = self.proxy_solution['x']

            feed_flows = [proxy_solution['feed_flows[0]'], proxy_solution['feed_flows[1]'],
                          proxy_solution['feed_flows[2]']]

            pool_flows = [proxy_solution['pool_flows[0]'], proxy_solution['pool_flows[1]'],
                          proxy_solution['pool_flows[2]'], proxy_solution['pool_flows[3]']
                          ]
            product_flows = [proxy_solution['product_flows[0]'], proxy_solution['product_flows[1]']]

            # The below calculates the compositions and standard deviations of compositions throughout the Haverly
            # network. Some handling must be done in case flows into a pool or product is zero (to avoid division
            # by zero errors)

            pool_mus = [0, 0]
            pool_stds = [0, 0]

            try:
                pool_mus[0] = (feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1]) / (
                        feed_flows[0] + feed_flows[1])
                pool_stds[0] = np.sqrt(
                    (feed_flows[0] ** 2 * feed_stds[0] ** 2 + feed_flows[1] ** 2 * feed_stds[1] ** 2) / (
                            feed_flows[0] + feed_flows[1]) ** 2)
            except:
                pool_mus[0] = 0
                pool_stds[0] = 1

            try:
                pool_mus[1] = feed_mus[2]
                pool_stds[1] = feed_stds[2]
            except:
                pool_mus[1] = 0
                pool_stds[1] = 1

            product_mus = [0, 0]
            product_stds = [0, 0]

            try:
                product_mus[0] = sum(pool_flows[i * 2 + 0] * pool_mus[i] for i in range(2)) / product_flows[0]
                product_stds[0] = np.sqrt(
                    sum(pool_flows[i * 2 + 0] ** 2 * pool_stds[i] ** 2 for i in range(2)) / product_flows[0] ** 2)
            except:
                product_mus[0] = 0
                product_stds[0] = 1

            try:
                product_mus[1] = sum(pool_flows[i * 2 + 1] * pool_mus[i] for i in range(2)) / product_flows[1]
                product_stds[1] = np.sqrt(
                    sum(pool_flows[i * 2 + 1] ** 2 * pool_stds[i] ** 2 for i in range(2)) / product_flows[1] ** 2)
            except:
                product_mus[1] = 0
                product_stds[1] = 1

            # Probabilities of the two products satisfying their quality constraints can be calculated using the
            # cumulative distribution function

            probabilities = [0, 0]

            probabilities[0] = norm_cdf(self.product_qualities[0], product_mus[0], product_stds[0])
            probabilities[1] = norm_cdf(self.product_qualities[1], product_mus[1], product_stds[1])

            self.stochastic_solution = (sum(feed_flows[i] * feed_price[i] for i in range(3)) -
                                        sum(probabilities[i] * product_flows[i] * product_price[i] for i in range(2)))
        elif self.problem_name == 'Haverly_3':
            feed_price = [6, 13, 10]
            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            # Load problem stats and data that will already have been set or obtained through optimisation

            feed_mus = self.feed_mus
            feed_stds = [self.std, self.std, self.std]

            proxy_solution = self.proxy_solution['x']

            feed_flows = [proxy_solution['feed_flows[0]'], proxy_solution['feed_flows[1]'],
                          proxy_solution['feed_flows[2]']]

            pool_flows = [proxy_solution['pool_flows[0]'], proxy_solution['pool_flows[1]'],
                          proxy_solution['pool_flows[2]'], proxy_solution['pool_flows[3]']
                          ]
            product_flows = [proxy_solution['product_flows[0]'], proxy_solution['product_flows[1]']]

            # The below calculates the compositions and standard deviations of compositions throughout the Haverly
            # network. Some handling must be done in case flows into a pool or product is zero (to avoid division
            # by zero errors)

            pool_mus = [0, 0]
            pool_stds = [0, 0]

            try:
                pool_mus[0] = (feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1]) / (
                        feed_flows[0] + feed_flows[1])
                pool_stds[0] = np.sqrt(
                    (feed_flows[0] ** 2 * feed_stds[0] ** 2 + feed_flows[1] ** 2 * feed_stds[1] ** 2) / (
                            feed_flows[0] + feed_flows[1]) ** 2)
            except:
                pool_mus[0] = 0
                pool_stds[0] = 1

            try:
                pool_mus[1] = feed_mus[2]
                pool_stds[1] = feed_stds[2]
            except:
                pool_mus[1] = 0
                pool_stds[1] = 1

            product_mus = [0, 0]
            product_stds = [0, 0]

            try:
                product_mus[0] = sum(pool_flows[i * 2 + 0] * pool_mus[i] for i in range(2)) / product_flows[0]
                product_stds[0] = np.sqrt(
                    sum(pool_flows[i * 2 + 0] ** 2 * pool_stds[i] ** 2 for i in range(2)) / product_flows[0] ** 2)
            except:
                product_mus[0] = 0
                product_stds[0] = 1

            try:
                product_mus[1] = sum(pool_flows[i * 2 + 1] * pool_mus[i] for i in range(2)) / product_flows[1]
                product_stds[1] = np.sqrt(
                    sum(pool_flows[i * 2 + 1] ** 2 * pool_stds[i] ** 2 for i in range(2)) / product_flows[1] ** 2)
            except:
                product_mus[1] = 0
                product_stds[1] = 1

            # Probabilities of the two products satisfying their quality constraints can be calculated using the
            # cumulative distribution function
            probabilities = [0, 0]

            probabilities[0] = norm_cdf(self.product_qualities[0], product_mus[0], product_stds[0])
            probabilities[1] = norm_cdf(self.product_qualities[1], product_mus[1], product_stds[1])

            self.stochastic_solution = (sum(feed_flows[i] * feed_price[i] for i in range(3)) -
                                        sum(probabilities[i] * product_flows[i] * product_price[i] for i in range(2)))
        elif self.problem_name == 'Foulds_2':
            feed_price = [6, 16, 10, 3, 13, 7]
            product_price = [9, 15, 6, 12]
            product_quality = [2.5, 1.5, 3, 2]

            # Load problem stats and data that will already have been set or obtained through optimisation

            feed_mus = self.feed_mus
            feed_stds = [self.std, self.std, self.std, self.std, self.std, self.std]

            proxy_solution = self.proxy_solution['x']

            feed_flows = [proxy_solution['feed_flows[0]'], proxy_solution['feed_flows[1]'],
                          proxy_solution['feed_flows[2]'], proxy_solution['feed_flows[3]'],
                          proxy_solution['feed_flows[4]'], proxy_solution['feed_flows[5]']]

            pool_flows = [proxy_solution['pool_flows[0]'], proxy_solution['pool_flows[1]'],
                          proxy_solution['pool_flows[2]'], proxy_solution['pool_flows[3]'],
                          proxy_solution['pool_flows[4]'], proxy_solution['pool_flows[5]'],
                          proxy_solution['pool_flows[6]'], proxy_solution['pool_flows[7]'],
                          proxy_solution['pool_flows[8]'], proxy_solution['pool_flows[9]'],
                          proxy_solution['pool_flows[10]'], proxy_solution['pool_flows[11]'],
                          proxy_solution['pool_flows[12]'], proxy_solution['pool_flows[13]'],
                          proxy_solution['pool_flows[14]'], proxy_solution['pool_flows[15]']
                          ]
            product_flows = [proxy_solution['product_flows[0]'], proxy_solution['product_flows[1]'],
                             proxy_solution['product_flows[2]'], proxy_solution['product_flows[3]']]

            # The below calculates the compositions and standard deviations of compositions throughout the Foulds 2
            # network. Some handling must be done in case flows into a pool or product is zero (to avoid division
            # by zero errors)

            pool_mus = [0, 0, 0, 0]
            pool_stds = [0, 0, 0, 0]

            try:
                pool_mus[0] = (feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1]) / (feed_flows[0] + feed_flows[1])
                pool_stds[0] = np.sqrt((feed_flows[0]**2 * feed_stds[0]**2 + feed_flows[1]**2 * feed_stds[1]**2)/(feed_flows[0] + feed_flows[1])**2)
            except:
                pool_mus[0] = 0
                pool_stds[0] = 1

            try:
                pool_mus[1] = feed_mus[2]
                pool_stds[1] = feed_stds[2]
            except:
                pool_mus[1] = 0
                pool_stds[1] = 1

            try:
                pool_mus[2] = (feed_flows[3] * feed_mus[3] + feed_flows[4] * feed_mus[4]) / (feed_flows[3] + feed_flows[4])
                pool_stds[2] = np.sqrt((feed_flows[3]**2 * feed_stds[3]**2 + feed_flows[4]**2 * feed_stds[4]**2)/(feed_flows[3] + feed_flows[4])**2)
            except:
                pool_mus[2] = 0
                pool_stds[2] = 1

            try:
                pool_mus[3] = feed_mus[5]
                pool_stds[3] = feed_stds[5]
            except:
                pool_mus[3] = 0
                pool_stds[3] = 1

            product_mus = [0, 0, 0, 0]
            product_stds = [0, 0, 0, 0]

            try:
                product_mus[0] = sum(pool_flows[i * 4 + 0] * pool_mus[i] for i in range(4)) / product_flows[0]
                product_stds[0] = np.sqrt(
                    sum(pool_flows[i * 4 + 0] ** 2 * pool_stds[i] ** 2 for i in range(4)) / product_flows[0] ** 2)
            except:
                product_mus[0] = 0
                product_stds[0] = 1

            try:
                product_mus[1] = sum(pool_flows[i * 4 + 1] * pool_mus[i] for i in range(4)) / product_flows[1]
                product_stds[1] = np.sqrt(
                    sum(pool_flows[i * 4 + 1] ** 2 * pool_stds[i] ** 2 for i in range(4)) / product_flows[1] ** 2)
            except:
                product_mus[1] = 0
                product_stds[1] = 1

            try:
                product_mus[2] = sum(pool_flows[i * 4 + 2] * pool_mus[i] for i in range(4)) / product_flows[2]
                product_stds[2] = np.sqrt(
                    sum(pool_flows[i * 4 + 2] ** 2 * pool_stds[i] ** 2 for i in range(4)) / product_flows[2] ** 2)
            except:
                product_mus[2] = 0
                product_stds[2] = 1

            try:
                product_mus[3] = sum(pool_flows[i * 4 + 3] * pool_mus[i] for i in range(4)) / product_flows[3]
                product_stds[3] = np.sqrt(
                    sum(pool_flows[i * 4 + 3] ** 2 * pool_stds[i] ** 2 for i in range(4)) / product_flows[3] ** 2)
            except:
                product_mus[3] = 0
                product_stds[3] = 1

            # Probabilities of the two products satisfying their quality constraints can be calculated using the
            # cumulative distribution function
            probabilities = [0, 0, 0, 0]

            probabilities[0] = norm_cdf(self.product_qualities[0], product_mus[0], product_stds[0])
            probabilities[1] = norm_cdf(self.product_qualities[1], product_mus[1], product_stds[1])
            probabilities[2] = norm_cdf(self.product_qualities[2], product_mus[2], product_stds[2])
            probabilities[3] = norm_cdf(self.product_qualities[3], product_mus[3], product_stds[3])

            self.stochastic_solution = (sum(feed_flows[i] * feed_price[i] for i in range(6)) -
                                   sum(probabilities[i] * product_flows[i] * product_price[i] for i in range(4)))
        elif self.problem_name == 'Segarwak':
            pass

    def save_results(self):
        """
        Since the structure of the resulting saving method depends quite strongly on the model used, this is simply a
        placeholder, meant to be overridden in the child classes.
        """
        pass


class ScenarioPooling(UncertainModel):
    """
    A class for creating and solving the uncertain pooling problems through a scenario approach (stochastic programming
    with discrete random variables). Includes a method for actually solving the problem along with a required method
    for creating the different scenarios used in the solution step
    """

    def __init__(self, problem_name, std=0.02, **kwargs):
        self.model_class = 'scenario'
        self.problem_name = problem_name
        self.std = std
        self.local_solver_tol = 1.0e-8  # Tolerance for Gurobi
        self.iteration_counter = 0
        self.problem_status = 'unsolved'
        if kwargs.get('scen_gen_strat', 'Lee') == 'Lee':
            self.scenario_generation_strategy = 'Lee' # Scenario generation strategy outlined in a 2010 paper by Lee
        elif kwargs.get('scen_gen_strat', None) == 'Basic':
            self.scenario_generation_strategy = 'Basic'
            self.scenario_generation_strategy_po = kwargs.get('scen_gen_strat_po', 0.3)
        self.num_scen = 3  ## Number of scenarios per uncertain variable
        self.gurobi_time_limit = 500 ## Time limit for Gurobi in seconds

    def solve_problem(self):
        """
        Method to solve the scenario proxy model. The model is also constructed in here. The main philosophy si to
        identify problem variables that are the same across scenarios and others that vary. For the ones that vary
        one "sub-variable" has to be defined per scenario and the relevant constraints also have to be defined per
        scenario.
        """

        # Select the problem based on the instance problem name

        if self.problem_name == 'Haverly_1':
            m = gp.Model('Haverly_1_discrete')
            self.feed_mus = [3, 1, 2]
            self.product_qualities = [2.5, 1.5]

            self.scenario_generation()  # Generate scenarios using the specified scenario generation method

            # Specify the relevant problem parameters

            feed_price = [6, 16, 10]

            product_price = [9, 15]
            product_quality = self.product_qualities  # Cleans up notation a bit further down.

            product_demand = [100, 200]

            # Set up the gurobi variables

            x_feed_flows = m.addVars(3, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(4, name='pool_flows', lb=0, ub=product_demand + product_demand)
            x_product_flows = m.addVars(2, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = [m.addVars(self.num_scen, self.num_scen, name='pool1_comps'),
                                   m.addVars(self.num_scen, name='pool2_comps')]

            x_product_compositions = m.addVars(2, self.num_scen, self.num_scen, self.num_scen, name='product_comps')
            # x_y are binary variables denoting whether the product quality constraints are satisfied in different
            # scenarios.
            x_y = m.addVars(2, self.num_scen, self.num_scen, self.num_scen, vtype=GRB.BINARY)
            # x_p are variables denoting the probability that a particular product stream satisfies the quality
            # constraints.
            x_p = m.addVars(2, lb=0, ub=1)

            m.update()

            # Set lower and upper bounds on the pool and  product compositions in different scenarios according to
            # basic interval arithmetics.
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            x_product_compositions[i, j1, j2, j3].LB = \
                                min(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                    self.var3_scenarios[j3])
                            x_product_compositions[i, j1, j2, j3].UB = \
                                max(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                    self.var3_scenarios[j3])

            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    x_pool_compositions[0][j1, j2].LB = \
                        min(self.var1_scenarios[j1], self.var2_scenarios[j2])
                    x_pool_compositions[0][j1, j2].UB = \
                        max(self.var1_scenarios[j1], self.var2_scenarios[j2])

            for j3 in range(self.num_scen):
                x_pool_compositions[1][j3].LB = \
                    self.var3_scenarios[j3]
                x_pool_compositions[1][j3].UB = \
                    self.var3_scenarios[j3]

            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(3)]) -
                           gp.quicksum([x_p[j] * x_product_flows[j] * product_price[j] for j in range(2)]))

            ## Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(2)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(2, 4)]))

            ## Pool component balance
            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    m.addConstr(x_feed_flows[0] * self.var1_scenarios[j1] + x_feed_flows[1] * self.var2_scenarios[j2] ==
                                gp.quicksum([x_pool_flows[i] * x_pool_compositions[0][j1, j2] for i in range(2)]))

            for j3 in range(self.num_scen):
                m.addConstr(self.var3_scenarios[j3] == x_pool_compositions[1][j3])

            ## Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 0] for i in range(2)]) == x_product_flows[0])

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 1] for i in range(2)]) == x_product_flows[1])

            ## Product quality balance

            # In contrast to the standard pooling problem, where the product quality simply restrict the product
            # quality, in the scenario approach we apply logical constraints to set the binary x_y variables to 1 if the
            # original constraint is satisfied and 0 otherwise
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            m.addConstr(x_pool_flows[0] * x_pool_compositions[0][j1, j2] + x_pool_flows[2] *
                                        x_pool_compositions[1][j3] ==
                                        x_product_flows[0] * x_product_compositions[0, j1, j2, j3])
                            m.addConstr(x_pool_flows[1] * x_pool_compositions[0][j1, j2] + x_pool_flows[3] *
                                        x_pool_compositions[1][j3] ==
                                        x_product_flows[1] * x_product_compositions[1, j1, j2, j3])

                            m.addConstr(x_product_compositions[0, j1, j2, j3] * x_y[0, j1, j2, j3] <=
                                        product_quality[0])
                            m.addConstr(x_product_compositions[1, j1, j2, j3] * x_y[1, j1, j2, j3] <=
                                        product_quality[1])
            prob_sums = [0, 0]

            # Set up constraints for the probabilities of quality satisfaction by summing the binary satisfaction
            # variables, weighted by the probabilities of each scenario.
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            prob_sums[i] += (self.var1_scenario_probs[j1] * self.var2_scenario_probs[j2] *
                                             self.var3_scenario_probs[j3]) * x_y[i, j1, j2, j3]
                m.addConstr(x_p[i] == prob_sums[i])
        elif self.problem_name == 'Haverly_2':
            m = gp.Model('Haverly_2_discrete')
            self.feed_mus = [3, 1, 2]
            self.product_qualities = [2.5, 1.5]

            self.scenario_generation()  # Generate scenarios using the specified scenario generation method

            # Specify the relevant problem parameters

            feed_price = [6, 16, 10]

            product_price = [9, 15]
            product_quality = self.product_qualities  # Cleans up notation a bit further down

            product_demand = [600, 200]

            # Set up the gurobi variables

            x_feed_flows = m.addVars(3, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(4, name='pool_flows', lb=0, ub=product_demand + product_demand)
            x_product_flows = m.addVars(2, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = [m.addVars(self.num_scen, self.num_scen, name='pool1_comps'),
                                   m.addVars(self.num_scen, name='pool2_comps')]

            x_product_compositions = m.addVars(2, self.num_scen, self.num_scen, self.num_scen, name='product_comps')

            # x_y are binary variables denoting whether the product quality constraints are satisfied in different
            # scenarios.
            x_y = m.addVars(2, self.num_scen, self.num_scen, self.num_scen, vtype=GRB.BINARY)

            # x_p are variables denoting the probability that a particular product stream satisfies the quality
            # constraints.
            x_p = m.addVars(2, lb=0, ub=1)

            m.update()

            # Set lower and upper bounds on the pool and  product compositions in different scenarios according to
            # basic interval arithmetics.
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            x_product_compositions[i, j1, j2, j3].LB = \
                                min(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                    self.var3_scenarios[j3])
                            x_product_compositions[i, j1, j2, j3].UB = \
                                max(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                    self.var3_scenarios[j3])

            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    x_pool_compositions[0][j1, j2].LB = \
                        min(self.var1_scenarios[j1], self.var2_scenarios[j2])
                    x_pool_compositions[0][j1, j2].UB = \
                        max(self.var1_scenarios[j1], self.var2_scenarios[j2])

            for j3 in range(self.num_scen):
                x_pool_compositions[1][j3].LB = \
                    self.var3_scenarios[j3]
                x_pool_compositions[1][j3].UB = \
                    self.var3_scenarios[j3]

            # Objective function
            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(3)]) -
                           gp.quicksum([x_p[j] * x_product_flows[j] * product_price[j] for j in range(2)]))

            # Pool mass balance
            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(2)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(2, 4)]))

            # Pool component balance
            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    m.addConstr(x_feed_flows[0] * self.var1_scenarios[j1] + x_feed_flows[1] * self.var2_scenarios[j2] ==
                                gp.quicksum([x_pool_flows[i] * x_pool_compositions[0][j1, j2] for i in range(2)]))

            for j3 in range(self.num_scen):
                m.addConstr(self.var3_scenarios[j3] == x_pool_compositions[1][j3])

            # Product mass balance
            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 0] for i in range(2)]) == x_product_flows[0])
            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 1] for i in range(2)]) == x_product_flows[1])

            # Product quality balance

            # In contrast to the standard pooling problem, where the product quality simply restrict the product
            # quality, in the scenario approach we apply logical constraints to set the binary x_y variables to 1 if the
            # original constraint is satisfied and 0 otherwise
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            m.addConstr(x_pool_flows[0] * x_pool_compositions[0][j1, j2] + x_pool_flows[2] *
                                        x_pool_compositions[1][j3] ==
                                        x_product_flows[0] * x_product_compositions[0, j1, j2, j3])
                            m.addConstr(x_pool_flows[1] * x_pool_compositions[0][j1, j2] + x_pool_flows[3] *
                                        x_pool_compositions[1][j3] ==
                                        x_product_flows[1] * x_product_compositions[1, j1, j2, j3])

                            m.addConstr(x_product_compositions[0, j1, j2, j3] * x_y[0, j1, j2, j3] <=
                                        product_quality[0])
                            m.addConstr(x_product_compositions[1, j1, j2, j3] * x_y[1, j1, j2, j3] <=
                                        product_quality[1])
            prob_sums = [0, 0]

            # Set up constraints for the probabilities of quality satisfaction by summing the binary satisfaction
            # variables, weighted by the probabilities of each scenario.
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            prob_sums[i] += (self.var1_scenario_probs[j1] * self.var2_scenario_probs[j2] *
                                             self.var3_scenario_probs[j3]) * x_y[i, j1, j2, j3]
                m.addConstr(x_p[i] == prob_sums[i])
        elif self.problem_name == 'Haverly_3':
            m = gp.Model('Haverly_3_discrete')
            self.feed_mus = [3, 1, 2]
            self.product_qualities = [2.5, 1.5]

            self.scenario_generation()  # Create the scenarios (values and probabilities) for the scenario model

            # Specify the relevant problem parameters
            feed_price = [6, 13, 10]

            product_price = [9, 15]
            product_quality = self.product_qualities  # Cleans up notation a bit further down

            product_demand = [100, 200]

            # Set up the gurobi variables
            x_feed_flows = m.addVars(3, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(4, name='pool_flows', lb=0, ub=product_demand + product_demand)
            x_product_flows = m.addVars(2, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = [m.addVars(self.num_scen, self.num_scen, name='pool1_comps'),
                                   m.addVars(self.num_scen, name='pool2_comps')]

            x_product_compositions = m.addVars(2, self.num_scen, self.num_scen, self.num_scen, name='product_comps')

            # x_y are binary variables denoting whether the product quality constraints are satisfied in different
            # scenarios.
            x_y = m.addVars(2, self.num_scen, self.num_scen, self.num_scen, vtype=GRB.BINARY)

            # x_p are variables denoting the probability that a particular product stream satisfies the quality
            # constraints.
            x_p = m.addVars(2, lb=0, ub=1)

            m.update()

            # Set lower and upper bounds on the pool and  product compositions in different scenarios according to
            # basic interval arithmetics.
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            x_product_compositions[i, j1, j2, j3].LB = \
                                min(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                    self.var3_scenarios[j3])
                            x_product_compositions[i, j1, j2, j3].UB = \
                                max(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                    self.var3_scenarios[j3])

            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    x_pool_compositions[0][j1, j2].LB = \
                        min(self.var1_scenarios[j1], self.var2_scenarios[j2])
                    x_pool_compositions[0][j1, j2].UB = \
                        max(self.var1_scenarios[j1], self.var2_scenarios[j2])

            for j3 in range(self.num_scen):
                x_pool_compositions[1][j3].LB = \
                    self.var3_scenarios[j3]
                x_pool_compositions[1][j3].UB = \
                    self.var3_scenarios[j3]

            # Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(3)]) -
                           gp.quicksum([x_p[j] * x_product_flows[j] * product_price[j] for j in range(2)]))

            # Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(2)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(2, 4)]))

            # Pool component balance
            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    m.addConstr(x_feed_flows[0] * self.var1_scenarios[j1] + x_feed_flows[1] * self.var2_scenarios[j2] ==
                                gp.quicksum([x_pool_flows[i] * x_pool_compositions[0][j1, j2] for i in range(2)]))

            for j3 in range(self.num_scen):
                m.addConstr(self.var3_scenarios[j3] == x_pool_compositions[1][j3])

            # Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 0] for i in range(2)]) == x_product_flows[0])
            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 1] for i in range(2)]) == x_product_flows[1])

            # Product quality balance

            # In contrast to the standard pooling problem, where the product quality simply restrict the product
            # quality, in the scenario approach we apply logical constraints to set the binary x_y variables to 1 if the
            # original constraint is satisfied and 0 otherwise
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            m.addConstr(x_pool_flows[0] * x_pool_compositions[0][j1, j2] + x_pool_flows[2] *
                                        x_pool_compositions[1][j3] ==
                                        x_product_flows[0] * x_product_compositions[0, j1, j2, j3])
                            m.addConstr(x_pool_flows[1] * x_pool_compositions[0][j1, j2] + x_pool_flows[3] *
                                        x_pool_compositions[1][j3] ==
                                        x_product_flows[1] * x_product_compositions[1, j1, j2, j3])

                            m.addConstr(x_product_compositions[0, j1, j2, j3] * x_y[0, j1, j2, j3] <=
                                        product_quality[0])
                            m.addConstr(x_product_compositions[1, j1, j2, j3] * x_y[1, j1, j2, j3] <=
                                        product_quality[1])
            prob_sums = [0, 0]

            # In contrast to the standard pooling problem, where the product quality simply restrict the product
            # quality, in the scenario approach we apply logical constraints to set the binary x_y variables to 1 if the
            # original constraint is satisfied and 0 otherwise
            for i in range(2):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            prob_sums[i] += (self.var1_scenario_probs[j1] * self.var2_scenario_probs[j2] *
                                             self.var3_scenario_probs[j3]) * x_y[i, j1, j2, j3]
                m.addConstr(x_p[i] == prob_sums[i])
        elif self.problem_name == 'Foulds_2':
            m = gp.Model('Foulds_2_discrete')
            self.feed_mus = [3, 1, 2, 3.5, 1.5, 2.5]
            self.product_qualities = [2.5, 1.5, 3, 2]

            self.scenario_generation()  # Create the scenarios (values and probabilities) for the scenario model

            # Specify the relevant model parameters
            feed_price = [6, 16, 10, 3, 13, 7]

            product_price = [9, 15, 6, 12]
            product_quality = self.product_qualities  # Cleans up notation further down

            product_demand = [100, 200, 100, 200]

            # Set up the gurobi variables
            x_feed_flows = m.addVars(6, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(16, name='pool_flows', lb=0,
                                     ub=product_demand + product_demand + product_demand + product_demand)
            x_product_flows = m.addVars(4, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = [m.addVars(self.num_scen, self.num_scen, name='pool1_comps'),
                                   m.addVars(self.num_scen, name='pool2_comps'),
                                   m.addVars(self.num_scen, self.num_scen, name='pool3_comps'),
                                   m.addVars(self.num_scen, name='pool4_comps')]

            x_product_compositions = m.addVars(4, self.num_scen, self.num_scen, self.num_scen, self.num_scen,
                                               self.num_scen, self.num_scen, name='product_comps')

            # x_y are binary variables denoting whether the product quality constraints are satisfied in different
            # scenarios.
            x_y = m.addVars(4, self.num_scen, self.num_scen, self.num_scen, self.num_scen, self.num_scen, self.num_scen,
                            vtype=GRB.BINARY)

            # x_p are variables denoting the probability that a particular product stream satisfies the quality
            # constraints.
            x_p = m.addVars(4, lb=0, ub=1)

            m.update()

            # Set lower and upper bounds on the pool and  product compositions in different scenarios according to
            # basic interval arithmetics.
            for i in range(4):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            for j4 in range(self.num_scen):
                                for j5 in range(self.num_scen):
                                    for j6 in range(self.num_scen):
                                        x_product_compositions[i, j1, j2, j3, j4, j5, j6].LB = \
                                            min(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                                self.var3_scenarios[j3],
                                                self.var4_scenarios[j4], self.var5_scenarios[j5],
                                                self.var6_scenarios[j6])
                                        x_product_compositions[i, j1, j2, j3, j4, j5, j6].UB = \
                                            max(self.var1_scenarios[j1], self.var2_scenarios[j2],
                                                self.var3_scenarios[j3],
                                                self.var4_scenarios[j4], self.var5_scenarios[j5],
                                                self.var6_scenarios[j6])
            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    x_pool_compositions[0][j1, j2].LB = \
                        min(self.var1_scenarios[j1], self.var2_scenarios[j2])
                    x_pool_compositions[0][j1, j2].UB = \
                        max(self.var1_scenarios[j1], self.var2_scenarios[j2])

            for j3 in range(self.num_scen):
                x_pool_compositions[1][j3].LB = \
                    self.var3_scenarios[j3]
                x_pool_compositions[1][j3].UB = \
                    self.var3_scenarios[j3]

            for j4 in range(self.num_scen):
                for j5 in range(self.num_scen):
                    x_pool_compositions[2][j4, j5].LB = \
                        min(self.var4_scenarios[j4], self.var5_scenarios[j5])
                    x_pool_compositions[2][j4, j5].UB = \
                        max(self.var4_scenarios[j4], self.var5_scenarios[j5])

            for j6 in range(self.num_scen):
                x_pool_compositions[3][j6].LB = \
                    self.var6_scenarios[j6]
                x_pool_compositions[3][j6].UB = \
                    self.var6_scenarios[j6]

            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(6)]) -
                           gp.quicksum([x_p[j] * x_product_flows[j] * product_price[j] for j in range(4)]))

            ## Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(0, 4)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(4, 8)]))
            m.addConstr(x_feed_flows[3] + x_feed_flows[4] == gp.quicksum([x_pool_flows[i] for i in range(8, 12)]))
            m.addConstr(x_feed_flows[5] == gp.quicksum([x_pool_flows[i] for i in range(12, 16)]))

            ## Pool component balance
            for j1 in range(self.num_scen):
                for j2 in range(self.num_scen):
                    m.addConstr(x_feed_flows[0] * self.var1_scenarios[j1] + x_feed_flows[1] * self.var2_scenarios[j2] ==
                                gp.quicksum([x_pool_flows[i] * x_pool_compositions[0][j1, j2] for i in range(0, 4)]))

            for j3 in range(self.num_scen):
                m.addConstr(self.var3_scenarios[j3] == x_pool_compositions[1][j3])

            for j4 in range(self.num_scen):
                for j5 in range(self.num_scen):
                    m.addConstr(x_feed_flows[3] * self.var4_scenarios[j4] + x_feed_flows[4] * self.var5_scenarios[j5] ==
                                gp.quicksum([x_pool_flows[i] * x_pool_compositions[2][j4, j5] for i in range(8, 12)]))

            for j6 in range(self.num_scen):
                m.addConstr(self.var6_scenarios[j6] == x_pool_compositions[3][j6])

            ## Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 0] for i in range(4)]) == x_product_flows[0])

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 1] for i in range(4)]) == x_product_flows[1])

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 2] for i in range(4)]) == x_product_flows[2])

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 3] for i in range(4)]) == x_product_flows[3])

            ## Product quality balance

            for i in range(4):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            for j4 in range(self.num_scen):
                                for j5 in range(self.num_scen):
                                    for j6 in range(self.num_scen):
                                        m.addConstr(x_pool_flows[0] * x_pool_compositions[0][j1, j2] + x_pool_flows[4] *
                                                    x_pool_compositions[1][j3] + x_pool_flows[8] *
                                                    x_pool_compositions[2][
                                                        j4, j5] + x_pool_flows[12] * x_pool_compositions[3][j6] ==
                                                    x_product_flows[0] * x_product_compositions[
                                                        0, j1, j2, j3, j4, j5, j6])
                                        m.addConstr(x_pool_flows[1] * x_pool_compositions[0][j1, j2] + x_pool_flows[5] *
                                                    x_pool_compositions[1][j3] + x_pool_flows[9] *
                                                    x_pool_compositions[2][
                                                        j4, j5] + x_pool_flows[13] * x_pool_compositions[3][j6] ==
                                                    x_product_flows[1] * x_product_compositions[
                                                        1, j1, j2, j3, j4, j5, j6])
                                        m.addConstr(x_pool_flows[2] * x_pool_compositions[0][j1, j2] + x_pool_flows[6] *
                                                    x_pool_compositions[1][j3] + x_pool_flows[10] *
                                                    x_pool_compositions[2][
                                                        j4, j5] + x_pool_flows[14] * x_pool_compositions[3][j6] ==
                                                    x_product_flows[2] * x_product_compositions[
                                                        2, j1, j2, j3, j4, j5, j6])
                                        m.addConstr(x_pool_flows[3] * x_pool_compositions[0][j1, j2] + x_pool_flows[7] *
                                                    x_pool_compositions[1][j3] + x_pool_flows[11] *
                                                    x_pool_compositions[2][
                                                        j4, j5] + x_pool_flows[15] * x_pool_compositions[3][j6] ==
                                                    x_product_flows[3] * x_product_compositions[
                                                        3, j1, j2, j3, j4, j5, j6])

                                        m.addConstr(x_product_compositions[0, j1, j2, j3, j4, j5, j6] * x_y[
                                            0, j1, j2, j3, j4, j5, j6] <=
                                                    product_quality[0])
                                        m.addConstr(x_product_compositions[1, j1, j2, j3, j4, j5, j6] * x_y[
                                            1, j1, j2, j3, j4, j5, j6] <=
                                                    product_quality[1])
                                        m.addConstr(x_product_compositions[2, j1, j2, j3, j4, j5, j6] * x_y[
                                            2, j1, j2, j3, j4, j5, j6] <=
                                                    product_quality[2])
                                        m.addConstr(x_product_compositions[3, j1, j2, j3, j4, j5, j6] * x_y[
                                            3, j1, j2, j3, j4, j5, j6] <=
                                                    product_quality[3])
            prob_sums = [0, 0, 0, 0]

            # In contrast to the standard pooling problem, where the product quality simply restrict the product
            # quality, in the scenario approach we apply logical constraints to set the binary x_y variables to 1 if the
            # original constraint is satisfied and 0 otherwise
            for i in range(4):
                for j1 in range(self.num_scen):
                    for j2 in range(self.num_scen):
                        for j3 in range(self.num_scen):
                            for j4 in range(self.num_scen):
                                for j5 in range(self.num_scen):
                                    for j6 in range(self.num_scen):
                                        prob_sums[i] += ((self.var1_scenario_probs[j1] * self.var2_scenario_probs[j2] *
                                                         self.var3_scenario_probs[j3] * self.var4_scenario_probs[j4] *
                                                         self.var5_scenario_probs[j5] * self.var6_scenario_probs[j6]) *
                                                         x_y[i, j1, j2, j3, j4, j5, j6])
                m.addConstr(x_p[i] == prob_sums[i])
        elif self.problem_name == 'Segarwak':
            """
            Based on the literature model of the gas extraction and distribution facilities in Segarwak
            """
            m = gp.Model('Segarwak')
            self.feed_mus = [0.72, 0.88, 0.27, 9.23, 3.41, 0.68, 1.64, 1.45, 8.85, 1.59, 1.59, 2.43, 0.95, 2.30, 3.34, 3.83, 3.83]
            self.product_qualities = [2.8, 2.8, 2.8]

            self.scenario_generation()

            num_feed_streams = 17
            num_product_streams = 3

            feed_price = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            product_price = [5364.17, 5364.17, 5364.17]
            product_quality = [2.8, 2.8, 2.8]

            product_demand = [1317, 2155, 2874]
            min_product_delivery = [838, 718, 718]

            x_feed_flows = m.addVars(17, name='feed_flows', lb=[48, 48, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     ub=[156, 150, 254, 400, 997, 456, 1137, 186, 480, 718, 718, 637, 977, 193, 2954,
                                         1078, 1078])
            x_platform_flows = m.addVars(7, name='platform_flows', lb=[48, 0, 0, 0, 0, 599, 0],
                                         ub=[305, 654, 1593, 718, 1807, 2634, 1078])
            x_t_flow = m.addVars(1, name='T_flow', lb=[0], ub=[2634])
            x_collector_flow = m.addVars(5, name='collector_flow', lb=[0, 0, 0, 0, 0], ub=[1676, 1137, 3592, 862, 2155])
            x_slug_catcher_flow = m.addVars(4, name='slug_catcher_flow', lb=[0, 0, 0, 0], ub=[1676, 2634, 958, 2155])
            x_product_flows = m.addVars(3, name='product_flows', lb=min_product_delivery, ub=product_demand)

            x_BYP_composition = m.addVars(1, name='BYP_composition', lb=[0.72], ub=[0.88])
            x_E11P_composition = m.addVars(1, name='E11P_composition', lb=[0.27], ub=[9.23])
            x_F23P_composition = m.addVars(1, name='F23P_composition', lb=[0.68], ub=[1.64])
            x_B11P_composition = m.addVars(1, name='B11P_compsition', lb=[1.59], ub=[8.85])
            x_M3P_copmosition = m.addVars(1, name='M3P_composition', lb=[0.95], ub=[2.43])
            x_M1P_compositions = m.addVars(self.num_scen, name='M1P_compositions')
            for i in range(self.num_scen):
                x_M1P_compositions[i].LB = min(3.83, self.var1_scenarios[i])
                x_M1P_compositions[i].UB = max(3.83, self.var1_scenarios[i])
            m.update()
            x_T_compositions = m.addVars(self.num_scen, name='T_compositions')
            for i in range(self.num_scen):
                x_T_compositions[i].LB = min(x_M3P_copmosition[0].LB, x_M1P_compositions[i].LB)
                x_T_compositions[i].UB = max(x_M3P_copmosition[0].UB, x_M1P_compositions[i].UB)

            x_E11RA_composition = m.addVars(1, name='E11RA_composition',
                                            lb=min(x_E11P_composition[0].LB, 3.41, x_F23P_composition[0].LB, 1.45),
                                            ub=max(x_E11P_composition[0].UB, 3.41, x_F23P_composition[0].UB, 1.45))
            m.update()

            x_E11RB_compositions = m.addVars(self.num_scen, name='E11RB_composition')
            for i in range(self.num_scen):
                x_E11RB_compositions[i].LB = min(x_E11RA_composition[0].LB, x_B11P_composition[0].LB, 1.59,
                                                x_T_compositions[i].LB)
                x_E11RB_compositions[i].UB = max(x_E11RA_composition[0].UB, x_B11P_composition[0].UB, 1.59,
                                                x_T_compositions[i].UB)

            m.update()

            x_E11RC_compositions = m.addVars(self.num_scen, name='E11RC_compositions')
            for i in range(self.num_scen):
                x_E11RC_compositions[i].LB = min(x_E11RB_compositions[i].LB, x_M1P_compositions[i].LB, 3.83)
                x_E11RC_compositions[i].UB = max(x_E11RB_compositions[i].UB, x_M1P_compositions[i].UB, 3.83)

            x_SC1_composition = m.addVars(1, name='SC1_composition',
                                          lb=min(x_BYP_composition[0].LB, x_E11RA_composition[0].LB),
                                          ub=max(x_BYP_composition[0].UB, x_E11RA_composition[0].UB))

            m.update()
            x_SC2_compositions = m.addVars(self.num_scen, name='SC2_compositions')
            for i in range(self.num_scen):
                x_SC2_compositions[i].LB = x_E11RB_compositions[i].LB
                x_SC2_compositions[i].UB = x_E11RB_compositions[i].UB

            x_SC3_compositions = m.addVars(self.num_scen, name='SC3_compositions')
            for i in range(self.num_scen):
                x_SC3_compositions[i].LB = x_E11RC_compositions[i].LB
                x_SC3_compositions[i].UB = x_E11RC_compositions[i].UB

            m.update()

            x_LNG1_composition = m.addVars(1, name='LNG1_composition', lb=x_SC1_composition[0].LB,
                                           ub=x_SC1_composition[0].UB)
            x_LNG2_compositions = m.addVars(self.num_scen, name='LNG2_compositions')
            for i in range(self.num_scen):
                x_LNG2_compositions[i].LB = x_SC2_compositions[i].LB
                x_LNG2_compositions[i].UB = x_SC2_compositions[i].UB

            x_LNG3_compositions = m.addVars(self.num_scen, name='LNG3_compositions')
            for i in range(self.num_scen):
                x_LNG3_compositions[i].LB = min(x_SC2_compositions[i].LB, x_SC3_compositions[i].LB)
                x_LNG3_compositions[i].UB = max(x_SC2_compositions[i].UB, x_SC3_compositions[i].UB)

            #x_platform_compositions = m.addVars(6, self.num_scen, name='platform_comps', lb=[0.72, 0.27, 0.68, 1.59, 0.95, -np.inf],
            #                                    ub=[0.88, 9.23, 1.64, 8.85, 2.43, np.inf])

            #x_t_compositions = m.addVars(self.num_scen, name='t_comps')

            x_collector_compositions = m.addVars(3, self.num_scen, name='collector_comps')

            x_slug_catcher_compositions = m.addVars(3, self.num_scen, name='slug_catcher_comps')

            x_product_compositions = m.addVars(3, self.num_scen, name='product_comps')

            y_feeds = m.addVars(7, name='feed_activation', vtype=GRB.BINARY)

            y_platform_activation = m.addVars(3, name='platform_activation', vtype=GRB.BINARY)

            y_t_activation = m.addVars(1, name='t_activation', vtype=GRB.BINARY)

            y_collector = m.addVars(2, name='collector_activation', vtype=GRB.BINARY)

            y_slug_catcher = m.addVars(2, name='slug_catcher_activation', vtype=GRB.BINARY)

            y_LNG = m.addVars(2, name='LNG_activation', vtype=GRB.BINARY)

            '''---------------------------audreyisthebest-----------------------'''

            y_product_satisfaction = m.addVars(3, self.num_scen, vtype=GRB.BINARY)

            x_p = m.addVars(3, lb=0, ub=1)

            m.update()

            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(num_feed_streams)]) -
                           gp.quicksum([x_p[j] * x_product_flows[j] * product_price[j] for j in range(num_product_streams)]))

            ## Platform mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == x_platform_flows[0])
            m.addConstr(x_feed_flows[2] + x_feed_flows[3] == x_platform_flows[1])
            m.addConstr(x_feed_flows[5] + x_feed_flows[6] == x_platform_flows[2])
            m.addConstr(x_feed_flows[8] + x_feed_flows[9] == x_platform_flows[3])
            m.addConstr(x_feed_flows[11] + x_feed_flows[12] + x_feed_flows[13] == x_platform_flows[4])
            m.addConstr(x_feed_flows[14] + x_feed_flows[15] == x_platform_flows[5] + x_platform_flows[6])

            ## T mass balance

            m.addConstr(x_platform_flows[4] + x_platform_flows[5] == x_t_flow[0])

            ## Collector mass balance

            m.addConstr(x_platform_flows[1] + x_feed_flows[4] + x_platform_flows[2] + x_feed_flows[7] ==
                        x_collector_flow[0] + x_collector_flow[1])

            m.addConstr(x_collector_flow[1] + x_platform_flows[3] + x_feed_flows[10] + x_t_flow[0] ==
                        x_collector_flow[2] + x_collector_flow[3])

            m.addConstr(x_collector_flow[3] + x_platform_flows[6] + x_feed_flows[16] ==
                        x_collector_flow[4])

            ## Slug catcher mass balance

            m.addConstr(x_platform_flows[0] + x_collector_flow[0] == x_slug_catcher_flow[0])

            m.addConstr(x_collector_flow[2] == x_slug_catcher_flow[1] + x_slug_catcher_flow[2])

            m.addConstr(x_collector_flow[4] == x_slug_catcher_flow[3])

            ## Product mass balance

            m.addConstr(x_slug_catcher_flow[0] == x_product_flows[0])

            m.addConstr(x_slug_catcher_flow[1] == x_product_flows[1])

            m.addConstr(x_slug_catcher_flow[2] + x_slug_catcher_flow[3] == x_product_flows[2])

            ## Platform composition balance

            m.addConstr(x_feed_flows[0] * self.feed_mus[0] + x_feed_flows[1] * self.feed_mus[1] ==
                        x_platform_flows[0] * x_BYP_composition[0])

            m.addConstr(x_feed_flows[2] * self.feed_mus[2] + x_feed_flows[3] * self.feed_mus[3] ==
                        x_platform_flows[1] * x_E11P_composition[0])

            m.addConstr(x_feed_flows[5] * self.feed_mus[5] + x_feed_flows[6] * self.feed_mus[6] ==
                        x_platform_flows[2] * x_F23P_composition[0])

            m.addConstr(x_feed_flows[8] * self.feed_mus[8] + x_feed_flows[9] * self.feed_mus[9] ==
                        x_platform_flows[3] * x_B11P_composition[0])

            m.addConstr(x_feed_flows[11] * self.feed_mus[11] + x_feed_flows[12] * self.feed_mus[12] +
                        x_feed_flows[13] * self.feed_mus[13] == x_platform_flows[4] * x_M3P_copmosition[0])

            for i in range(self.num_scen):
                m.addConstr(x_feed_flows[14] * self.var1_scenarios[i] + x_feed_flows[15] * self.feed_mus[15] ==
                            (x_platform_flows[5] + x_platform_flows[6]) * x_M1P_compositions[i])

            ## T composition balance
            for i in range(self.num_scen):
                m.addConstr(x_platform_flows[4] * x_M3P_copmosition[0] + x_platform_flows[5] * x_M1P_compositions[i] ==
                            x_t_flow[0] * x_T_compositions[i])

            ## Collector composition balance

            m.addConstr(x_platform_flows[1] * x_E11P_composition[0] + x_feed_flows[4] * self.feed_mus[4] +
                        x_platform_flows[2] * x_F23P_composition[0] + x_feed_flows[7] * self.feed_mus[7] ==
                        (x_collector_flow[0] + x_collector_flow[1]) * x_E11RA_composition[0])

            for i in range(self.num_scen):
                m.addConstr(x_collector_flow[1] * x_E11RA_composition[0] + x_platform_flows[3] * x_B11P_composition[0] +
                            x_feed_flows[10] * self.feed_mus[10] + x_t_flow[0] * x_T_compositions[i] ==
                            (x_collector_flow[2] + x_collector_flow[3]) * x_E11RB_compositions[i])

            for i in range(self.num_scen):
                m.addConstr(x_collector_flow[4] * x_E11RB_compositions[i] + x_platform_flows[6] * x_M1P_compositions[i]
                            + x_feed_flows[16] * self.feed_mus[16] == x_collector_flow[4] * x_E11RC_compositions[i])

            ## Slug-catcher composition balance

            m.addConstr(x_platform_flows[0] * x_BYP_composition[0] + x_collector_flow[0] * x_E11RA_composition[0] ==
                        x_slug_catcher_flow[0] * x_SC1_composition[0])

            for i in range(self.num_scen):
                m.addConstr(x_collector_flow[2] * x_E11RB_compositions[i] ==
                            (x_slug_catcher_flow[1] + x_slug_catcher_flow[2]) * x_SC2_compositions[i])

            for i in range(self.num_scen):
                m.addConstr(x_E11RC_compositions[i] == x_SC3_compositions[i])

            ## Product composition balance

            m.addConstr(x_SC1_composition[0] == x_LNG1_composition[0])

            for i in range(self.num_scen):
                m.addConstr(x_SC2_compositions[i] == x_LNG2_compositions[i])

            for i in range(self.num_scen):
                m.addConstr(x_slug_catcher_flow[2] * x_SC2_compositions[i] + x_slug_catcher_flow[3] *
                            x_SC3_compositions[i] == x_product_flows[2] * x_LNG3_compositions[i])

            ## Product quality satisfaction

            for i in range(self.num_scen):
                m.addConstr(x_LNG1_composition[0]  <= self.product_qualities[0])
                m.addConstr(x_LNG2_compositions[i] * y_product_satisfaction[1, i] <= self.product_qualities[1])
                m.addConstr(x_LNG3_compositions[i] * y_product_satisfaction[2, i] <= self.product_qualities[2])

            prob_sums = [0, 0, 0]

            x_p[0] = 1

            for i in range(1, 3):
                for j in range(self.num_scen):
                    prob_sums[i] += self.var1_scenario_probs[j] * y_product_satisfaction[i, j]
                m.addConstr(x_p[i] == prob_sums[i])

        m.params.NonConvex = 2  # Required parameter update for Gurobi to solve nonconvex problems
        m.params.TimeLimit = self.gurobi_time_limit
        m.optimize()

        self.scenario_incumbent = m.objVal  # The current best obtained solution
        self.scenario_bound = m.ObjBound  # The best lower bound achieved. May differ from the best obtained solution.
        self.runTime = m.Runtime
        self.problem_status = 'solved'

        solution_dict = {}

        for var in m.getVars():
            solution_dict[var.VarName] = var.X

        self.proxy_solution = {}
        self.proxy_solution['x'] = solution_dict

    def scenario_generation(self):
        """
        A method to construct scenarios used for the scenario proxy model. Two main options are already included:
        One by Lee(2010) and a basic one matching the variance of the discrete model with the variance of the true
        underlying random variable
        """
        if self.scenario_generation_strategy == 'Lee':
            if self.problem_name == 'Haverly_1':
                # First we define the values the uncertain parameters will take in the different scenarios
                self.var1_scenarios = [
                    (-3 * self.std + self.feed_mus[0] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]
                self.var2_scenarios = [
                    (-3 * self.std + self.feed_mus[1] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]
                self.var3_scenarios = [
                    (-3 * self.std + self.feed_mus[2] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]

                # Next we determine the probabilities associated with each realisation of the uncertain parameters

                self.var1_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std / self.num_scen, loc=self.feed_mus[0],
                             scale=self.std)]
                self.var2_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std / self.num_scen, loc=self.feed_mus[1],
                             scale=self.std)]
                self.var3_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std / self.num_scen, loc=self.feed_mus[2],
                             scale=self.std)]

                for i in range(1, self.num_scen - 1):
                    self.var1_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[0], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[0] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[0], scale=self.std))

                    self.var2_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[1], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[1] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[1], scale=self.std))

                    self.var3_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[2], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[2] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[2], scale=self.std))


                self.var1_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[0], scale=self.std))
                self.var2_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[1], scale=self.std))
                self.var3_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[2], scale=self.std))

                self.p_o = 'Lee'  # If the scenario generation method is 'Lee' we find it easier to identify it as such
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Haverly_2':
                # First we define the values different uncertain parameters can take in the different scenarios
                self.var1_scenarios = [
                    (-3 * self.std + self.feed_mus[0] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]
                self.var2_scenarios = [
                    (-3 * self.std + self.feed_mus[1] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]
                self.var3_scenarios = [
                    (-3 * self.std + self.feed_mus[2] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]

                # Next we specify the probabilities associated with the different realisations of the uncertain params
                self.var1_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std / self.num_scen, loc=self.feed_mus[0],
                             scale=self.std)]
                self.var2_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std / self.num_scen, loc=self.feed_mus[1],
                             scale=self.std)]
                self.var3_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std / self.num_scen, loc=self.feed_mus[2],
                             scale=self.std)]

                for i in range(1, self.num_scen - 1):
                    self.var1_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[0], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[0] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[0], scale=self.std))

                    self.var2_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[1], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[1] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[1], scale=self.std))

                    self.var3_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[2], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[2] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[2], scale=self.std))

                self.var1_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[0], scale=self.std))
                self.var2_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[1], scale=self.std))
                self.var3_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[2], scale=self.std))

                self.p_o = 'Lee'  # We find it easier to identify the 'Lee' option like this
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Haverly_3':
                # Start by defining the values the uncertain parameters can take in different scenarios
                self.var1_scenarios = [
                    (-3 * self.std + self.feed_mus[0] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]
                self.var2_scenarios = [
                    (-3 * self.std + self.feed_mus[1] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]
                self.var3_scenarios = [
                    (-3 * self.std + self.feed_mus[2] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]

                # Then specify the probabilities associated with the different scenarios
                self.var1_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std / self.num_scen, loc=self.feed_mus[0],
                             scale=self.std)]
                self.var2_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std / self.num_scen, loc=self.feed_mus[1],
                             scale=self.std)]
                self.var3_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std / self.num_scen, loc=self.feed_mus[2],
                             scale=self.std)]

                for i in range(1, self.num_scen - 1):
                    self.var1_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[0], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[0] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[0], scale=self.std))

                    self.var2_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[1], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[1] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[1], scale=self.std))

                    self.var3_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[2], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[2] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[2], scale=self.std))

                self.var1_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[0], scale=self.std))
                self.var2_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[1], scale=self.std))
                self.var3_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[2], scale=self.std))

                self.p_o = 'Lee'
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Foulds_2':
                # First we define the values the uncertain parameters can take in different scenarios
                self.var1_scenarios = [(-3 * self.std + self.feed_mus[0] + 3 * self.std / self.num_scen +
                                        i * 6 * self.std / self.num_scen) for i in range(self.num_scen)]
                self.var2_scenarios = [(-3 * self.std + self.feed_mus[1] + 3 * self.std / self.num_scen +
                                        i * 6 * self.std / self.num_scen) for i in range(self.num_scen)]
                self.var3_scenarios = [(-3 * self.std + self.feed_mus[2] + 3 * self.std / self.num_scen +
                                        i * 6 * self.std / self.num_scen) for i in range(self.num_scen)]
                self.var4_scenarios = [(-3 * self.std + self.feed_mus[3] + 3 * self.std / self.num_scen +
                                        i * 6 * self.std / self.num_scen) for i in range(self.num_scen)]
                self.var5_scenarios = [(-3 * self.std + self.feed_mus[4] + 3 * self.std / self.num_scen +
                                        i * 6 * self.std / self.num_scen) for i in range(self.num_scen)]
                self.var6_scenarios = [(-3 * self.std + self.feed_mus[5] + 3 * self.std / self.num_scen +
                                        i * 6 * self.std / self.num_scen) for i in range(self.num_scen)]

                # Next specify the probabilities associated with each scenario
                self.var1_scenario_probs = [norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std / self.num_scen,
                                                     loc=self.feed_mus[0], scale=self.std)]
                self.var2_scenario_probs = [norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std / self.num_scen,
                                                     loc=self.feed_mus[1], scale=self.std)]
                self.var3_scenario_probs = [norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std / self.num_scen,
                                                     loc=self.feed_mus[2], scale=self.std)]
                self.var4_scenario_probs = [norm.cdf(-3 * self.std + self.feed_mus[3] + 6 * self.std / self.num_scen,
                                                     loc=self.feed_mus[3], scale=self.std)]
                self.var5_scenario_probs = [norm.cdf(-3 * self.std + self.feed_mus[4] + 6 * self.std / self.num_scen,
                                                     loc=self.feed_mus[4], scale=self.std)]
                self.var6_scenario_probs = [norm.cdf(-3 * self.std + self.feed_mus[5] + 6 * self.std / self.num_scen,
                                                     loc=self.feed_mus[5], scale=self.std)]

                for i in range(1, self.num_scen - 1):
                    self.var1_scenario_probs.append(norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * (i + 1) /
                                                             self.num_scen, loc=self.feed_mus[0], scale=self.std) -
                                                    norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std * i /
                                                             self.num_scen, loc=self.feed_mus[0], scale=self.std))

                    self.var2_scenario_probs.append(norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * (i + 1) /
                                                             self.num_scen, loc=self.feed_mus[1], scale=self.std) -
                                                    norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std * i /
                                                             self.num_scen, loc=self.feed_mus[1], scale=self.std))

                    self.var3_scenario_probs.append(norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * (i + 1) /
                                                             self.num_scen, loc=self.feed_mus[2], scale=self.std) -
                                                    norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std * i /
                                                             self.num_scen, loc=self.feed_mus[2], scale=self.std))

                    self.var4_scenario_probs.append(norm.cdf(-3 * self.std + self.feed_mus[3] + 6 * self.std * (i + 1) /
                                                             self.num_scen, loc=self.feed_mus[3], scale=self.std) -
                                                    norm.cdf(-3 * self.std + self.feed_mus[3] + 6 * self.std * i /
                                                             self.num_scen, loc=self.feed_mus[3], scale=self.std))

                    self.var5_scenario_probs.append(norm.cdf(-3 * self.std + self.feed_mus[4] + 6 * self.std * (i + 1) /
                                                             self.num_scen, loc=self.feed_mus[4], scale=self.std) -
                                                    norm.cdf(-3 * self.std + self.feed_mus[4] + 6 * self.std * i /
                                                             self.num_scen, loc=self.feed_mus[4], scale=self.std))

                    self.var6_scenario_probs.append(norm.cdf(-3 * self.std + self.feed_mus[5] + 6 * self.std * (i + 1) /
                                                             self.num_scen, loc=self.feed_mus[5], scale=self.std) -
                                                    norm.cdf(-3 * self.std + self.feed_mus[5] + 6 * self.std * i /
                                                             self.num_scen, loc=self.feed_mus[5], scale=self.std))

                self.var1_scenario_probs.append(1 - norm.cdf(-3 * self.std + self.feed_mus[0] + 6 * self.std *
                                                             (self.num_scen - 1) / self.num_scen, loc=self.feed_mus[0],
                                                             scale=self.std))
                self.var2_scenario_probs.append(1 - norm.cdf(-3 * self.std + self.feed_mus[1] + 6 * self.std *
                                                             (self.num_scen - 1) / self.num_scen, loc=self.feed_mus[1],
                                                             scale=self.std))
                self.var3_scenario_probs.append(1 - norm.cdf(-3 * self.std + self.feed_mus[2] + 6 * self.std *
                                                             (self.num_scen - 1) / self.num_scen, loc=self.feed_mus[2],
                                                             scale=self.std))
                self.var4_scenario_probs.append(1 - norm.cdf(-3 * self.std + self.feed_mus[3] + 6 * self.std *
                                                             (self.num_scen - 1) / self.num_scen, loc=self.feed_mus[3],
                                                             scale=self.std))
                self.var5_scenario_probs.append(1 - norm.cdf(-3 * self.std + self.feed_mus[4] + 6 * self.std *
                                                             (self.num_scen - 1) / self.num_scen, loc=self.feed_mus[4],
                                                             scale=self.std))
                self.var6_scenario_probs.append(1 - norm.cdf(-3 * self.std + self.feed_mus[5] + 6 * self.std *
                                                             (self.num_scen - 1) / self.num_scen, loc=self.feed_mus[5],
                                                             scale=self.std))

                self.p_o = 'Lee'
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Segarwak':
                self.var1_scenarios = [
                    (-3 * self.std + self.feed_mus[14] + 3 * self.std / self.num_scen + i * 6 * self.std / self.num_scen)
                    for i in range(self.num_scen)]

                self.var1_scenario_probs = [
                    norm.cdf(-3 * self.std + self.feed_mus[14] + 6 * self.std / self.num_scen, loc=self.feed_mus[14],
                             scale=self.std)]
                for i in range(1, self.num_scen - 1):
                    self.var1_scenario_probs.append(
                        norm.cdf(-3 * self.std + self.feed_mus[14] + 6 * self.std * (i + 1) / self.num_scen,
                                 loc=self.feed_mus[14], scale=self.std) - norm.cdf(-3 * self.std + self.feed_mus[14] +
                                                                                  6 * self.std * i / self.num_scen,
                                                                                  loc=self.feed_mus[14], scale=self.std))
                self.var1_scenario_probs.append(
                    1 - norm.cdf(-3 * self.std + self.feed_mus[14] + 6 * self.std * (self.num_scen - 1) /
                                 self.num_scen, loc=self.feed_mus[14], scale=self.std))

                self.p_o = 'Lee'
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
        elif self.scenario_generation_strategy == 'Basic':
            if self.problem_name == 'Haverly_1':
                p_o = self.scenario_generation_strategy_po  # Probability for the low and high scenarios
                delta_x = self.std / (np.sqrt(2 * p_o))  # Deviation from the mean calculated to retain variance

                # Values and associated probabilities calculated for all scenarios
                self.var1_scenarios = [self.feed_mus[0] - delta_x, self.feed_mus[0], self.feed_mus[0] + delta_x]
                self.var1_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var2_scenarios = [self.feed_mus[1] - delta_x, self.feed_mus[1], self.feed_mus[1] + delta_x]
                self.var2_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var3_scenarios = [self.feed_mus[2] - delta_x, self.feed_mus[2], self.feed_mus[2] + delta_x]
                self.var3_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]

                self.p_o = p_o
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Haverly_2':
                p_o = self.scenario_generation_strategy_po  # Probability for the low and high scenarios
                delta_x = self.std / (np.sqrt(2 * p_o))  # Deviation from the mean calculated to retain variance

                # Values and associated probabilities calculated for all scenarios
                self.var1_scenarios = [self.feed_mus[0] - delta_x, self.feed_mus[0], self.feed_mus[0] + delta_x]
                self.var1_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var2_scenarios = [self.feed_mus[1] - delta_x, self.feed_mus[1], self.feed_mus[1] + delta_x]
                self.var2_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var3_scenarios = [self.feed_mus[2] - delta_x, self.feed_mus[2], self.feed_mus[2] + delta_x]
                self.var3_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]

                self.p_o = p_o
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Haverly_3':
                p_o = self.scenario_generation_strategy_po  # Probability for the low and high scenarios
                delta_x = self.std / (np.sqrt(2 * p_o))  # Deviation from the mean calculated to retain variance

                # Values and associated probabilities calculated for all scenarios
                self.var1_scenarios = [self.feed_mus[0] - delta_x, self.feed_mus[0], self.feed_mus[0] + delta_x]
                self.var1_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var2_scenarios = [self.feed_mus[1] - delta_x, self.feed_mus[1], self.feed_mus[1] + delta_x]
                self.var2_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var3_scenarios = [self.feed_mus[2] - delta_x, self.feed_mus[2], self.feed_mus[2] + delta_x]
                self.var3_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]

                self.p_o = p_o ## This has kind of already been set?
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Foulds_2':
                p_o = self.scenario_generation_strategy_po  # Probability for the low and high scenarios
                delta_x = self.std / (np.sqrt(2 * p_o))  # Deviation from the mean calculated to retain variance

                # Values and associated probabilities calculated for all scenarios
                self.var1_scenarios = [self.feed_mus[0] - delta_x, self.feed_mus[0], self.feed_mus[0] + delta_x]
                self.var1_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var2_scenarios = [self.feed_mus[1] - delta_x, self.feed_mus[1], self.feed_mus[1] + delta_x]
                self.var2_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var3_scenarios = [self.feed_mus[2] - delta_x, self.feed_mus[2], self.feed_mus[2] + delta_x]
                self.var3_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var4_scenarios = [self.feed_mus[3] - delta_x, self.feed_mus[3], self.feed_mus[3] + delta_x]
                self.var4_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var5_scenarios = [self.feed_mus[4] - delta_x, self.feed_mus[4], self.feed_mus[4] + delta_x]
                self.var5_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]
                self.var6_scenarios = [self.feed_mus[5] - delta_x, self.feed_mus[5], self.feed_mus[5] + delta_x]
                self.var6_scenario_probs = [p_o, 1.0 - 2 * p_o, p_o]

                self.p_o = p_o
                self.delta_x = self.var1_scenarios[1] - self.var1_scenarios[0]
            elif self.problem_name == 'Segarwak':
                pass

    def save_results(self):
        """
        A method to save the results to a csv result file. First the method checks if a results file for the appropriate
         problem/solution approach exists, and if it does it pull it and only updates the field for the relevant
         combination of true uncertainty variance and scenario generation choices. It then saves the file. If no
         appropriate file exists initially, it creates one from scratch.
        """
        try:
            dataframe = pd.read_csv(self.problem_name + '_scenario_results.csv', index_col=['std', 'p_o'])
        except:
            index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u'std', u'p_o'])
            dataframe = pd.DataFrame(index=index, columns=['delta_x', 'f_hat', 'f', 'f_LB', 'CPU'])

        dataframe.loc[(self.std, self.p_o), :] = [self.delta_x, self.scenario_incumbent, self.stochastic_solution,
                                                  self.scenario_bound, self.runTime]

        dataframe = dataframe.sort_index(ascending=False)

        dataframe.to_csv(self.problem_name + '_scenario_results.csv')


class RobustPooling(UncertainModel):
    """
    A class for creating and solving the uncertain pooling problems through a robust optimisation approach. Includes a
    method for solving the problem. Since the uncertainty set used here is based on the infinity norm, the worst-case
    situation arises when all product mus are at their upper bound and the problem can consequently be solved as a
    normal bilinear programming problem.
    """

    def __init__(self, problem_name, std=0.02, **kwargs):
        self.model_class = 'robust'
        self.problem_name = problem_name
        self.std = std
        self.local_solver_tol = 1.0e-8 # Gurobi tolerance
        self.iteration_counter = 0
        self.problem_status = 'unsolved'
        self.robust_radius = kwargs.get('r', 0.1) # Radius of the uncertainty set. Can be given as a parameter
        self.gurobi_time_limit = 500 ## Gurobi time limit in seconds

    def solve_problem(self):
        """
        Method to solve the robust proxy model. The model is also constructed in here. The main philosophy is to
        recognise that with an infinity norm induced uncertainty set, the worst-case is realised when all uncertain
        feed qualities are at their (mean + uncertainty radius). The nominal value for the feed qualities can then be
        updated accordingly, and the model be solved as a standard pooling problem.
        """
        if self.problem_name == 'Haverly_1':
            m = gp.Model('Haverly_1_discrete')
            self.feed_mus = [3, 1, 2]
            self.robust_feed_mus = [feed_mu + self.robust_radius for feed_mu in self.feed_mus]
            self.product_qualities = [2.5, 1.5]

            feed_price = [6, 16, 10]

            product_price = [9, 15]
            product_quality = [2.5, 1.5] ## Use class variable?

            product_demand = [100, 200]

            x_feed_flows = m.addVars(3, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(4, name='pool_flows', lb=0,
                                     ub=product_demand + product_demand)
            x_product_flows = m.addVars(2, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = m.addVars(2, name='pool1_comps')

            m.update()

            x_pool_compositions[0].LB = min(self.robust_feed_mus[0], self.robust_feed_mus[1])
            x_pool_compositions[0].UB = max(self.robust_feed_mus[0], self.robust_feed_mus[1])

            x_pool_compositions[1].LB = self.robust_feed_mus[2]
            x_pool_compositions[1].UB = self.robust_feed_mus[2]

            m.update()
            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(3)]) -
                           gp.quicksum([x_product_flows[j] * product_price[j] for j in range(2)]))

            ## Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(2)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(2, 4)]))

            ## Pool component balance
            m.addConstr(x_feed_flows[0] * self.robust_feed_mus[0] + x_feed_flows[1] * self.robust_feed_mus[1] ==
                        gp.quicksum([x_pool_flows[i] * x_pool_compositions[0] for i in range(2)]))

            m.addConstr(self.robust_feed_mus[2] == x_pool_compositions[1])

            ## Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 0] for i in range(2)]) == x_product_flows[0])

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 1] for i in range(2)]) == x_product_flows[1])


            ## Product quality balance

            m.addConstr(x_pool_flows[0] * x_pool_compositions[0] + x_pool_flows[2] *
                        x_pool_compositions[1] <= x_product_flows[0] * product_quality[0])
            m.addConstr(x_pool_flows[1] * x_pool_compositions[0] + x_pool_flows[3] *
                        x_pool_compositions[1] <= x_product_flows[1] * product_quality[1])
        elif self.problem_name == 'Haverly_2':
            m = gp.Model('Haverly_2_discrete')
            self.feed_mus = [3, 1, 2]
            self.robust_feed_mus = [feed_mu + self.robust_radius for feed_mu in self.feed_mus]
            self.product_qualities = [2.5, 1.5]

            feed_price = [6, 16, 10]

            product_price = [9, 15]
            product_quality = [2.5, 1.5]  ## Use class variable?

            product_demand = [600, 200]

            x_feed_flows = m.addVars(3, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(4, name='pool_flows', lb=0,
                                     ub=product_demand + product_demand)
            x_product_flows = m.addVars(2, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = m.addVars(2, name='pool1_comps')

            m.update()

            x_pool_compositions[0].LB = min(self.robust_feed_mus[0], self.robust_feed_mus[1])
            x_pool_compositions[0].UB = max(self.robust_feed_mus[0], self.robust_feed_mus[1])

            x_pool_compositions[1].LB = self.robust_feed_mus[2]
            x_pool_compositions[1].UB = self.robust_feed_mus[2]

            m.update()
            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(3)]) -
                           gp.quicksum([x_product_flows[j] * product_price[j] for j in range(2)]))

            ## Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(2)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(2, 4)]))

            ## Pool component balance
            m.addConstr(x_feed_flows[0] * self.robust_feed_mus[0] + x_feed_flows[1] * self.robust_feed_mus[1] ==
                        gp.quicksum([x_pool_flows[i] * x_pool_compositions[0] for i in range(2)]))

            m.addConstr(self.robust_feed_mus[2] == x_pool_compositions[1])

            ## Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 0] for i in range(2)]) == x_product_flows[0])

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 1] for i in range(2)]) == x_product_flows[1])

            ## Product quality balance

            m.addConstr(x_pool_flows[0] * x_pool_compositions[0] + x_pool_flows[2] *
                        x_pool_compositions[1] <= x_product_flows[0] * product_quality[0])
            m.addConstr(x_pool_flows[1] * x_pool_compositions[0] + x_pool_flows[3] *
                        x_pool_compositions[1] <= x_product_flows[1] * product_quality[1])
        elif self.problem_name == 'Haverly_3':
            m = gp.Model('Haverly_3_discrete')
            self.feed_mus = [3, 1, 2]
            self.robust_feed_mus = [feed_mu + self.robust_radius for feed_mu in self.feed_mus]
            self.product_qualities = [2.5, 1.5]

            feed_price = [6, 13, 10]

            product_price = [9, 15]
            product_quality = [2.5, 1.5]  ## Use class variable?

            product_demand = [100, 200]

            x_feed_flows = m.addVars(3, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(4, name='pool_flows', lb=0,
                                     ub=product_demand + product_demand)
            x_product_flows = m.addVars(2, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = m.addVars(2, name='pool1_comps')

            m.update()

            x_pool_compositions[0].LB = min(self.robust_feed_mus[0], self.robust_feed_mus[1])
            x_pool_compositions[0].UB = max(self.robust_feed_mus[0], self.robust_feed_mus[1])

            x_pool_compositions[1].LB = self.robust_feed_mus[2]
            x_pool_compositions[1].UB = self.robust_feed_mus[2]

            m.update()
            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(3)]) -
                           gp.quicksum([x_product_flows[j] * product_price[j] for j in range(2)]))

            ## Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(2)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(2, 4)]))

            ## Pool component balance
            m.addConstr(x_feed_flows[0] * self.robust_feed_mus[0] + x_feed_flows[1] * self.robust_feed_mus[1] ==
                        gp.quicksum([x_pool_flows[i] * x_pool_compositions[0] for i in range(2)]))

            m.addConstr(self.robust_feed_mus[2] == x_pool_compositions[1])

            ## Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 0] for i in range(2)]) == x_product_flows[0])

            m.addConstr(gp.quicksum([x_pool_flows[i * 2 + 1] for i in range(2)]) == x_product_flows[1])

            ## Product quality balance

            m.addConstr(x_pool_flows[0] * x_pool_compositions[0] + x_pool_flows[2] *
                        x_pool_compositions[1] <= x_product_flows[0] * product_quality[0])
            m.addConstr(x_pool_flows[1] * x_pool_compositions[0] + x_pool_flows[3] *
                        x_pool_compositions[1] <= x_product_flows[1] * product_quality[1])
        elif self.problem_name == 'Foulds_2':
            m = gp.Model('Foulds_2_discrete')
            self.feed_mus = [3, 1, 2, 3.5, 1.5, 2.5]
            self.robust_feed_mus = [feed_mu + self.robust_radius for feed_mu in self.feed_mus]
            self.product_qualities = [2.5, 1.5, 3, 2]

            feed_price = [6, 16, 10, 3, 13, 7]

            product_price = [9, 15, 6, 12]
            product_quality = [2.5, 1.5, 3, 2] ## Use class variable?

            product_demand = [100, 200, 100, 200]

            x_feed_flows = m.addVars(6, name='feed_flows', lb=0, ub=sum(product_demand))
            x_pool_flows = m.addVars(16, name='pool_flows', lb=0, ub=product_demand+product_demand+product_demand+product_demand)
            x_product_flows = m.addVars(4, name='product_flows', lb=0, ub=product_demand)
            x_pool_compositions = m.addVars(4, name='pool1_comps')

            m.update()

            x_pool_compositions[0].LB = min(self.robust_feed_mus[0], self.robust_feed_mus[1])
            x_pool_compositions[0].UB = max(self.robust_feed_mus[0], self.robust_feed_mus[1])

            x_pool_compositions[1].LB = self.robust_feed_mus[2]
            x_pool_compositions[1].UB = self.robust_feed_mus[2]

            x_pool_compositions[2].LB = min(self.robust_feed_mus[3], self.robust_feed_mus[4])
            x_pool_compositions[2].UB = max(self.robust_feed_mus[3], self.robust_feed_mus[4])

            x_pool_compositions[3].LB = self.robust_feed_mus[5]
            x_pool_compositions[3].UB = self.robust_feed_mus[5]

            m.update()
            ## Objective function

            m.setObjective(gp.quicksum([x_feed_flows[i] * feed_price[i] for i in range(6)]) -
                           gp.quicksum([x_product_flows[j] * product_price[j] for j in range(4)]))


            ## Pool mass balance

            m.addConstr(x_feed_flows[0] + x_feed_flows[1] == gp.quicksum([x_pool_flows[i] for i in range(0, 4)]))
            m.addConstr(x_feed_flows[2] == gp.quicksum([x_pool_flows[i] for i in range(4, 8)]))
            m.addConstr(x_feed_flows[3] + x_feed_flows[4] == gp.quicksum([x_pool_flows[i] for i in range(8, 12)]))
            m.addConstr(x_feed_flows[5] == gp.quicksum([x_pool_flows[i] for i in range(12, 16)]))

            ## Pool component balance
            m.addConstr(x_feed_flows[0] * self.robust_feed_mus[0] + x_feed_flows[1] * self.robust_feed_mus[1] ==
                        gp.quicksum([x_pool_flows[i] * x_pool_compositions[0] for i in range(0, 4)]))

            m.addConstr(self.robust_feed_mus[2] == x_pool_compositions[1])


            m.addConstr(x_feed_flows[3] * self.robust_feed_mus[3] + x_feed_flows[4] *self.robust_feed_mus[4] ==
                        gp.quicksum([x_pool_flows[i] * x_pool_compositions[2] for i in range(8, 12)]))

            m.addConstr(self.robust_feed_mus[5] == x_pool_compositions[3])

            ## Product mass balance

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 0] for i in range(4)]) == x_product_flows[0])

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 1] for i in range(4)]) == x_product_flows[1])

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 2] for i in range(4)]) == x_product_flows[2])

            m.addConstr(gp.quicksum([x_pool_flows[i * 4 + 3] for i in range(4)]) == x_product_flows[3])

            ## Product quality balance

            m.addConstr(x_pool_flows[0] * x_pool_compositions[0] + x_pool_flows[4] *
                        x_pool_compositions[1] + x_pool_flows[8] * x_pool_compositions[2] + x_pool_flows[12] *
                        x_pool_compositions[3] <= x_product_flows[0] * product_quality[0])
            m.addConstr(x_pool_flows[1] * x_pool_compositions[0] + x_pool_flows[5] *
                        x_pool_compositions[1] + x_pool_flows[9] * x_pool_compositions[2] + x_pool_flows[13] *
                        x_pool_compositions[3] <= x_product_flows[1] * product_quality[1])
            m.addConstr(x_pool_flows[2] * x_pool_compositions[0] + x_pool_flows[6] *
                        x_pool_compositions[1] + x_pool_flows[10] * x_pool_compositions[2] + x_pool_flows[14] *
                        x_pool_compositions[3] <=
                        x_product_flows[2] * product_quality[2])
            m.addConstr(x_pool_flows[3] * x_pool_compositions[0] + x_pool_flows[7] *
                        x_pool_compositions[1] + x_pool_flows[11] * x_pool_compositions[2] + x_pool_flows[15] *
                        x_pool_compositions[3] <=
                        x_product_flows[3] * product_quality[3])
        elif self.problem_name == 'Segarwak':
            pass

        m.params.TimeLimit = self.gurobi_time_limit
        m.params.NonConvex = 2 ## Required parameter update for Gurobi to solve nonconvex bilinear programming problems
        m.optimize()

        self.robust_incumbent = m.objVal # Best obtained solution
        self.robust_bound = m.ObjBound # Best remaining bound. If it is equal to the incumbent, global optimisation is guaranteed
        self.runTime = m.Runtime
        self.problem_status = 'solved'

        solution_dict = {}

        for var in m.getVars():
            solution_dict[var.VarName] = var.X

        self.proxy_solution = {}
        self.proxy_solution['x'] = solution_dict

    def save_results(self):
        """
        A method to save the results to a csv result file. First the method checks if a results file for the appropriate
         problem/solution approach exists, and if it does it pull it and only updates the field for the relevant
         combination of true uncertainty variance and scenario generation choices. It then saves the file. If no
         appropriate file exists initially, it creates one from scratch.
        """
        try:
            dataframe = pd.read_csv(self.problem_name + '_robust_results.csv', index_col=['std', 'r'])
        except:
            index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u'std', u'r'])
            dataframe = pd.DataFrame(index=index, columns=['f_hat', 'f', 'f_LB', 'CPU'])

        dataframe.loc[(self.std, self.robust_radius), :] = [self.robust_incumbent, self.stochastic_solution,
                                                            self.robust_bound, self.runTime]

        dataframe = dataframe.sort_index(ascending=False)

        dataframe.to_csv(self.problem_name + '_robust_results.csv')


class StochPooling(UncertainModel):
    """
    A class for creating and solving the uncertain pooling problem through a stochastic programming approach. The
    stochastic programming with normally distributed respects the true nature of the assumed uncertainty. Includes
    a method for solving the problem, that further relies on methods for lower bounding and upper bounding the problem
    within sub-regions of the original feasible region. Also includes a method for setting up the data structures needed
    for the spatial branch and bound and a method for lower bounding the error function (needed for the cumulative
    distribution function of a normally distributed random variable. As long as the solution here is feasible there
    should be no difference between the objective function outputted by the optimisation solver and what would have
    been returned by the ss_evaluator() method of the parent class.
    """
    def __init__(self, problem_name, std=0.02):
        self.model_class = 'continuous'
        self.problem_name = problem_name
        self.std = std

        # Solver cut offs are set in 1) iterations for partitioning the product compostition mu, 2) iterations for the
        # feed and pool flow rates and 3) runtime in seconds
        self.cut_offs = [30, np.inf, 50]
        self.local_solver_tol = 1.0e-8
        self.iteration_counter = 0
        self.problem_status = 'unsolved'

        #self.load_problem()

        self.set_up_global_structure()

    def set_up_global_structure(self):
        """
        Method to define many of the computational structures needed to keep track of the progress of the global solver
        and the original problem bounds
        """

        self.lower_bounds = []  # A list of lists of lower bounds
        self.upper_bounds = []  # A list of lists of upper bounds
        self.obj_lower_bound = []  # A list of lower bounds
        self.obj_upper_bound = []  # A list of upper bounds

        self.current_best_solution = np.inf  # The best solution found so far
        self.current_best_solution_x = None  # The variables values that give this solution
        self.iterational_best_lower_bound = []  # The best lower bound at each iteration

        if self.problem_name == 'Haverly_1':
            self.lower_bounds.append(np.array([0, 0, 0,
                                               0, 0, 0, 0,
                                               0, 0,
                                               1, 2,
                                               np.sqrt(0.5 * self.std ** 2), self.std,
                                               1, 1,
                                               np.sqrt(0.25 * 0.5 * self.std ** 2), np.sqrt(0.25 * 0.5 * self.std ** 2),
                                               0, 0]))
            self.upper_bounds.append(np.array([300, 300, 300,
                                               100, 200, 100, 200,
                                               100, 200,
                                               3, 2,
                                               self.std, self.std,
                                               2.5, 1.5,
                                               self.std, self.std,
                                               1, 1]))
        elif self.problem_name == 'Haverly_2':
            self.lower_bounds.append(np.array([0, 0, 0,
                                               0, 0, 0, 0,
                                               0, 0,
                                               1, 2,
                                               np.sqrt(0.5 * self.std ** 2), self.std,
                                               1, 1,
                                               np.sqrt(0.25 * 0.5 * self.std ** 2), np.sqrt(0.25 * 0.5 * self.std ** 2),
                                               0, 0]))
            self.upper_bounds.append(np.array([800, 800, 800,
                                               600, 200, 600, 200,
                                               600, 200,
                                               3, 2,
                                               self.std, self.std,
                                               2.5, 1.5,
                                               self.std, self.std,
                                               1, 1]))
        elif self.problem_name == 'Haverly_3':
            self.lower_bounds.append(np.array([0, 0, 0,
                                               0, 0, 0, 0,
                                               0, 0,
                                               1, 2,
                                               np.sqrt(0.5 * self.std ** 2), self.std,
                                               1, 1,
                                               np.sqrt(0.25 * 0.5 * self.std ** 2), np.sqrt(0.25 * 0.5 * self.std ** 2),
                                               0, 0]))
            self.upper_bounds.append(np.array([300, 300, 300,
                                               100, 200, 100, 200,
                                               100, 200,
                                               3, 2,
                                               self.std, self.std,
                                               2.5, 1.5,
                                               self.std, self.std,
                                               1, 1]))
        elif self.problem_name == 'Foulds_2':
            self.lower_bounds.append(np.array([0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0,
                                               1, 2, 1.5, 2.5,
                                               np.sqrt(0.5 * self.std ** 2), self.std, np.sqrt(0.5 * self.std ** 2),
                                               self.std,
                                               1, 1, 1, 1,
                                               np.sqrt(0.25 * 0.5 * self.std ** 2), np.sqrt(0.25 * 0.5 * self.std ** 2),
                                               np.sqrt(0.25 * 0.5 * self.std ** 2), np.sqrt(0.25 * 0.5 * self.std ** 2),
                                               0, 0, 0, 0]))
            self.upper_bounds.append(np.array([600, 600, 600, 600, 600, 600,
                                               100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200, 100, 200,
                                               100, 200,
                                               100, 200, 100, 200,
                                               3, 2, 3.5, 2.5,
                                               self.std, self.std, self.std, self.std,
                                               2.5, 1.5, 3.0, 2.0,
                                               self.std, self.std, self.std, self.std,
                                               1, 1, 1, 1]))
        elif self.problem_name == 'Segarwak':
            pass

    def solve_problem(self):
        """
        Method to solve the stochastic programming problem. The model is also constructed in here. The main philosophy
        is to construct the problem exactly as we best understand it using continuous (normally distributed) random
        variables. This means we are trying to optimise the model we are planning to validate the solution against,
        so successful optimisation here guarantees the best achievable solution for our wider problem.
        """
        start = time.time()

        problem_solved = False

        cut_offs = self.cut_offs

        if self.problem_name in ['Haverly_1', 'Haverly_2', 'Haverly_3']:
            self.feed_mus = [3, 1, 2]
            self.product_qualities = [2.5, 1.5]

            # Follow a general spatial branch and bound algorithmic framework to solve the nonconvex problem

            # Solve the initial lower bounding scheme to get a valid lower bound on the original problem
            candidate_lower_bound = self.lower_bounding(0)

            if candidate_lower_bound[0] is True:
                self.obj_lower_bound.append(candidate_lower_bound[1])

                self.iterational_best_lower_bound.append(candidate_lower_bound[1])

                candidate_upper_bound = self.upper_bounding(0)
                if candidate_upper_bound[0] is True:
                    self.obj_upper_bound.append(candidate_upper_bound[1])
                    if candidate_upper_bound[1] < self.current_best_solution:
                        self.current_best_solution = candidate_upper_bound[1]
                        self.current_best_solution_x = candidate_upper_bound[2]
                        self.runTimetoSol = time.time() - start
                    else:
                        pass
                else:
                    self.obj_upper_bound.append('Unconverged')

            self.outer_iterations = 1
            while not problem_solved:
                current_iteration_best_lb = np.inf
                partition_bound_preset = False
                if self.outer_iterations <= cut_offs[0]:
                    if self.outer_iterations % 2 == 0:
                        variable_to_partition = 13
                        product_quality = 2.5
                    else:
                        variable_to_partition = 14
                        product_quality = 1.5

                    index_to_partition = 0

                    for i in range(len(self.lower_bounds)):
                        if self.upper_bounds[i][variable_to_partition] > product_quality + 1.0e-5 and \
                                self.lower_bounds[i][
                                    variable_to_partition] < product_quality - 1.0e-5:
                            index_to_partition = i
                            partition_bound_preset = True
                            partition_bound = product_quality
                            break
                        elif self.upper_bounds[i][variable_to_partition] - self.lower_bounds[i][variable_to_partition] > \
                                self.upper_bounds[index_to_partition][variable_to_partition] - \
                                self.lower_bounds[index_to_partition][
                                    variable_to_partition]:
                            index_to_partition = i

                    lb_workset = self.lower_bounds[index_to_partition].copy()
                    ub_workset = self.upper_bounds[index_to_partition].copy()
                else:
                    index_to_partition = self.obj_lower_bound.index(min(self.obj_lower_bound))
                    lb_workset = self.lower_bounds[index_to_partition].copy()
                    ub_workset = self.upper_bounds[index_to_partition].copy()
                    variable_to_partition = np.argmax(ub_workset - lb_workset)

                del self.lower_bounds[index_to_partition]
                del self.upper_bounds[index_to_partition]
                del self.obj_lower_bound[index_to_partition]
                del self.obj_upper_bound[index_to_partition]

                if not partition_bound_preset:
                    partition_bound = (ub_workset[variable_to_partition] + lb_workset[variable_to_partition]) / 2

                new_lb_1 = lb_workset.copy()
                new_ub_1 = ub_workset.copy()

                new_lb_2 = lb_workset.copy()
                new_ub_2 = ub_workset.copy()

                new_ub_1[variable_to_partition] = partition_bound
                new_lb_2[variable_to_partition] = partition_bound

                self.lower_bounds.append(new_lb_1)
                self.lower_bounds.append(new_lb_2)

                self.upper_bounds.append(new_ub_1)
                self.upper_bounds.append(new_ub_2)

                candidate_lower_bound_1 = self.lower_bounding(-2)

                if candidate_lower_bound_1[0] is True and candidate_lower_bound_1[1] <= self.current_best_solution:
                    self.obj_lower_bound.append(candidate_lower_bound_1[1])

                    candidate_upper_bound_1 = self.upper_bounding(-2)
                    if candidate_upper_bound_1[0] is True:
                        self.obj_upper_bound.append(candidate_upper_bound_1[1])
                        if candidate_upper_bound_1[1] < self.current_best_solution:
                            self.current_best_solution = candidate_upper_bound_1[1]
                            self.current_best_solution_x = candidate_upper_bound_1[2]
                            self.runTimetoSol = time.time() - start
                    else:
                        self.obj_upper_bound.append('Unconverged')
                else:
                    del self.lower_bounds[-2]
                    del self.upper_bounds[-2]

                candidate_lower_bound_2 = self.lower_bounding(-1)

                if candidate_lower_bound_2[0] is True and candidate_lower_bound_2[1] <= self.current_best_solution:
                    self.obj_lower_bound.append(candidate_lower_bound_2[1])

                    candidate_upper_bound_2 = self.upper_bounding(-1)
                    if candidate_upper_bound_2[0] is True:
                        self.obj_upper_bound.append(candidate_upper_bound_2[1])
                        if candidate_upper_bound_2[1] < self.current_best_solution:
                            self.current_best_solution = candidate_upper_bound_2[1]
                            self.current_best_solution_x = candidate_upper_bound_2[2]
                            self.runTimetoSol = time.time() - start
                    else:
                        self.obj_upper_bound.append('Unconverged')
                else:
                    del self.lower_bounds[-1]
                    del self.upper_bounds[-1]

                convergence_criteria = False

                if (isinstance(self.current_best_solution,
                               float) and not self.obj_lower_bound) or self.current_best_solution - min(
                    self.obj_lower_bound) <= 1.0e-4:
                    convergence_criteria = True

                self.outer_iterations += 1
                print('---------------------------------------------------')
                print('Just finished the ', self.outer_iterations, ' iteration!')
                print('---------------------------------------------------')

                if self.outer_iterations == cut_offs[1] or time.time() - start >= cut_offs[2] or convergence_criteria:

                    end = time.time()

                    self.runTime = end - start

                    indices_to_delete = [i for i, v in enumerate(self.obj_upper_bound) if isinstance(v, str)]

                    if not self.obj_lower_bound:
                        self.best_lb = self.current_best_solution
                    else:
                        self.continuous_bound = float(min(self.obj_lower_bound))

                    self.continuous_solution = self.current_best_solution

                    for index in sorted(indices_to_delete, reverse=True):
                        del self.lower_bounds[index]
                        del self.upper_bounds[index]
                        del self.obj_lower_bound[index]
                        del self.obj_upper_bound[index]

                    print('List of all lower bounds: ', self.obj_lower_bound)
                    print('List of all upper bounds: ', self.obj_upper_bound)
                    print('Best solution objective: ', self.current_best_solution)
                    print('And the solution: ', self.current_best_solution_x)
                    print('The single best lower bound: ', self.iterational_best_lower_bound[-1])

                    self.problem_status = 'solved'

                    break
                else:
                    self.iterational_best_lower_bound.append(min(self.obj_lower_bound))

                    indices_to_delete = [i for i, v in enumerate(self.obj_lower_bound) if
                                         v >= self.current_best_solution]

                    for index in sorted(indices_to_delete, reverse=True):
                        del self.lower_bounds[index]
                        del self.upper_bounds[index]
                        del self.obj_lower_bound[index]
                        del self.obj_upper_bound[index]
        elif self.problem_name == 'Foulds_2':
            self.feed_mus = [3, 1, 2, 3.5, 1.5, 2.5]
            self.product_qualities = [2.5, 1.5, 3, 2]
            candidate_lower_bound = self.lower_bounding(0)

            if candidate_lower_bound[0] is True:
                self.obj_lower_bound.append(candidate_lower_bound[1])

                self.iterational_best_lower_bound.append(candidate_lower_bound[1])

                candidate_upper_bound = self.upper_bounding(0)
                if candidate_upper_bound[0] is True:
                    self.obj_upper_bound.append(candidate_upper_bound[1])
                    if candidate_upper_bound[1] < self.current_best_solution:
                        self.current_best_solution = candidate_upper_bound[1]
                        self.current_best_solution_x = candidate_upper_bound[2]
                        self.runTimetoSol = time.time() - start
                    else:
                        pass
                else:
                    self.obj_upper_bound.append('Unconverged')

            self.outer_iterations = 1
            while not problem_solved:
                current_iteration_best_lb = np.inf
                partition_bound_preset = False
                if self.outer_iterations <= cut_offs[0]:
                    if self.outer_iterations % 4 == 0:
                        variable_to_partition = 34
                        product_quality = 2.5
                    elif self.outer_iterations % 4 == 1:
                        variable_to_partition = 35
                        product_quality = 1.5
                    elif self.outer_iterations % 4 == 2:
                        variable_to_partition = 36
                        product_quality = 3
                    else:
                        variable_to_partition = 37
                        product_quality = 2

                    index_to_partition = 0

                    for i in range(len(self.lower_bounds)):
                        if self.upper_bounds[i][variable_to_partition] > product_quality + 1.0e-5 and self.lower_bounds[i][
                            variable_to_partition] < product_quality - 1.0e-5:
                            index_to_partition = i
                            partition_bound_preset = True
                            partition_bound = product_quality
                            break
                        elif self.upper_bounds[i][variable_to_partition] - self.lower_bounds[i][variable_to_partition] > \
                                self.upper_bounds[index_to_partition][variable_to_partition] - \
                                self.lower_bounds[index_to_partition][
                                    variable_to_partition]:
                            index_to_partition = i

                    lb_workset = self.lower_bounds[index_to_partition].copy()
                    ub_workset = self.upper_bounds[index_to_partition].copy()
                else:
                    index_to_partition = self.obj_lower_bound.index(min(self.obj_lower_bound))
                    lb_workset = self.lower_bounds[index_to_partition].copy()
                    ub_workset = self.upper_bounds[index_to_partition].copy()
                    variable_to_partition = np.argmax(ub_workset - lb_workset)

                del self.lower_bounds[index_to_partition]
                del self.upper_bounds[index_to_partition]
                del self.obj_lower_bound[index_to_partition]
                del self.obj_upper_bound[index_to_partition]

                if not partition_bound_preset:
                    partition_bound = (ub_workset[variable_to_partition] + lb_workset[variable_to_partition]) / 2

                new_lb_1 = lb_workset.copy()
                new_ub_1 = ub_workset.copy()

                new_lb_2 = lb_workset.copy()
                new_ub_2 = ub_workset.copy()

                new_ub_1[variable_to_partition] = partition_bound
                new_lb_2[variable_to_partition] = partition_bound

                self.lower_bounds.append(new_lb_1)
                self.lower_bounds.append(new_lb_2)

                self.upper_bounds.append(new_ub_1)
                self.upper_bounds.append(new_ub_2)

                candidate_lower_bound_1 = self.lower_bounding(-2)

                if candidate_lower_bound_1[0] is True and candidate_lower_bound_1[1] <= self.current_best_solution:
                    self.obj_lower_bound.append(candidate_lower_bound_1[1])

                    candidate_upper_bound_1 = self.upper_bounding(-2)
                    if candidate_upper_bound_1[0] is True:
                        self.obj_upper_bound.append(candidate_upper_bound_1[1])
                        if candidate_upper_bound_1[1] < self.current_best_solution:
                            self.current_best_solution = candidate_upper_bound_1[1]
                            self.current_best_solution_x = candidate_upper_bound_1[2]
                            self.runTimetoSol = time.time() - start
                    else:
                        self.obj_upper_bound.append('Unconverged')
                else:
                    del self.lower_bounds[-2]
                    del self.upper_bounds[-2]

                candidate_lower_bound_2 = self.lower_bounding(-1)

                if candidate_lower_bound_2[0] is True and candidate_lower_bound_2[1] <= self.current_best_solution:
                    self.obj_lower_bound.append(candidate_lower_bound_2[1])

                    candidate_upper_bound_2 = self.upper_bounding(-1)
                    if candidate_upper_bound_2[0] is True:
                        self.obj_upper_bound.append(candidate_upper_bound_2[1])
                        if candidate_upper_bound_2[1] < self.current_best_solution:
                            self.current_best_solution = candidate_upper_bound_2[1]
                            self.current_best_solution_x = candidate_upper_bound_2[2]
                            self.runTimetoSol = time.time() - start
                    else:
                        self.obj_upper_bound.append('Unconverged')
                else:
                    del self.lower_bounds[-1]
                    del self.upper_bounds[-1]

                convergence_criteria = False

                if (isinstance(self.current_best_solution,
                               float) and not self.obj_lower_bound) or self.current_best_solution - min(
                        self.obj_lower_bound) <= 1.0e-4:
                    convergence_criteria = True

                self.outer_iterations += 1
                print('---------------------------------------------------')
                print('Just finished the ', self.outer_iterations, ' iteration!')
                print('---------------------------------------------------')

                if self.outer_iterations == cut_offs[1] or time.time() - start >= cut_offs[2] or convergence_criteria:

                    end = time.time()

                    self.runTime = end - start

                    indices_to_delete = [i for i, v in enumerate(self.obj_upper_bound) if isinstance(v, str)]

                    if not self.obj_lower_bound:
                        self.best_lb = self.current_best_solution
                    else:
                        self.continuous_bound = float(min(self.obj_lower_bound))

                    self.continuous_solution = self.current_best_solution

                    for index in sorted(indices_to_delete, reverse=True):
                        del self.lower_bounds[index]
                        del self.upper_bounds[index]
                        del self.obj_lower_bound[index]
                        del self.obj_upper_bound[index]

                    print('List of all lower bounds: ', self.obj_lower_bound)
                    print('List of all upper bounds: ', self.obj_upper_bound)
                    print('Best solution objective: ', self.current_best_solution)
                    print('And the solution: ', self.current_best_solution_x)
                    print('The single best lower bound: ', self.iterational_best_lower_bound[-1])

                    self.problem_status = 'solved'

                    break
                else:
                    self.iterational_best_lower_bound.append(min(self.obj_lower_bound))

                    indices_to_delete = [i for i, v in enumerate(self.obj_lower_bound) if v >= self.current_best_solution]

                    for index in sorted(indices_to_delete, reverse=True):
                        del self.lower_bounds[index]
                        del self.upper_bounds[index]
                        del self.obj_lower_bound[index]
                        del self.obj_upper_bound[index]
        elif self.problem_name == 'Segarwak':
            pass

    def lower_bounding(self, node):
        """
        Method to solve a lower bounding problem for the stochastic programming approach. This creates a convex
        relaxation of the nonconvex problem and solves this to obtain a valid lower bound on the region of interest
        """
        if self.problem_name == 'Haverly_1' or self.problem_name == 'Haverly_2':

            # Set up the casadi variables needed for the problem
            feed_flows = casadi.SX.sym('feed_flows', 3)
            pool_flows = casadi.SX.sym('pool_flows', 4)
            product_flows = casadi.SX.sym('product_flows', 2)
            pool_mus = casadi.SX.sym('pool_mus', 2)
            pool_sigmas = casadi.SX.sym('pool_sigmas', 2)
            product_mus = casadi.SX.sym('product_mus', 2)
            product_sigmas = casadi.SX.sym('product_sigma', 2)
            product_p = casadi.SX.sym('product_probabilities', 2)

            # Specify the problem parameters
            feed_price = [6, 16, 10]
            feed_mus = [3, 1, 2]

            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            #self.update_bounds(node)

            lb = list(self.lower_bounds[node].copy())
            ub = list(self.upper_bounds[node].copy())

            # Set the objective function
            objective = casadi.sum1(feed_flows * feed_price) - casadi.sum1(product_p * product_flows * product_price)

            # Initiate a casadi SX object, and two lists to gradually build the constraints of the problem
            g = casadi.SX()
            lbg = []
            ubg = []

            # Pool mass balance
            g = casadi.vertcat(g,
                               feed_flows[0] + feed_flows[1] - casadi.sum1(pool_flows[:2]),
                               feed_flows[2] - casadi.sum1(pool_flows[2:4]),
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mass balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 2 for i in range(2)]]) - product_flows[0],
                               casadi.sum1(pool_flows[[i * 2 + 1 for i in range(2)]]) - product_flows[1],
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool mu balance
            pool_mu_bilinears = casadi.SX.sym('pool_mu_bilinears', 4)

            g = casadi.vertcat(g,
                               feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1] - casadi.sum1(
                                   pool_mu_bilinears[:2]),
                               feed_flows[2] * feed_mus[2] - casadi.sum1(pool_mu_bilinears[2:4])
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Next we have to define a number of artificial problem variables in order to obtain bilinear terms
            # that we can more easily define the convex relaxations for

            pool_sigma_bilinears = casadi.SX.sym('pool_sigma_bilinears', 4)
            pool_sigma_squared_bilinears = casadi.SX.sym('pool_sigma_squared_bilinears', 4)
            pool_sigma_bilinears_sums = casadi.SX.sym('pool_sigma_bilinears_sums', 2)
            pool_sigma_bilinears_sums_squared = casadi.SX.sym('pool_sigma_bilinears_sums_squared', 2)

            feed_flows_squared = casadi.SX.sym('feed_flows_squared', 3)

            # Pool sigma balance
            g = casadi.vertcat(g,
                               pool_sigma_bilinears_sums[0] - casadi.sum1(pool_sigma_bilinears[:2]),
                               pool_sigma_bilinears_sums[1] - casadi.sum1(pool_sigma_bilinears[2:4])
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool sum standard deviations
            g = casadi.vertcat(g,
                               feed_flows_squared[0] * self.std ** 2 + feed_flows_squared[1] * self.std ** 2 -
                               pool_sigma_bilinears_sums_squared[0],
                               feed_flows_squared[2] * self.std ** 2 - pool_sigma_bilinears_sums_squared[1],
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mu balance
            product_mu_bilinears = casadi.SX.sym('product_mu_bilinears', 2)

            g = casadi.vertcat(g,
                               casadi.sum1(pool_mu_bilinears[[i * 2 for i in range(2)]]) - product_mu_bilinears[0],
                               casadi.sum1(pool_mu_bilinears[[i * 2 + 1 for i in range(2)]]) - product_mu_bilinears[1]
                               )
            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product sigma balance

            product_sigma_bilinears = casadi.SX.sym('product_sigma_bilienars', 2)
            product_sigma_squared_bilinears = casadi.SX.sym('product_sigma_squared_bilinears', 2)

            g = casadi.vertcat(g,
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 2 for i in range(2)]]) -
                               product_sigma_squared_bilinears[0],
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 2 + 1 for i in range(2)]]) -
                               product_sigma_squared_bilinears[1]
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Finally the cumulative distribution function can be relaxed using over and under esttimators
            for i in range(2):
                if ub[13 + i] <= product_quality[i]:
                    g = casadi.vertcat(g,
                                       product_p[i] - 0.5 * (1 + casadi.erf(
                                           (product_quality[i] - product_mus[i]) / (product_sigmas[i] * np.sqrt(2))))
                                       )
                    lbg.append(-np.inf)
                    ubg.append(0)
                else:
                    first_intersect, first_slope = \
                        self.Get_erf_LB(product_quality[i], [lb[13 + i], ub[13 + i]], [lb[13 + i], ub[13 + i]])[1]

                    g = casadi.vertcat(g,
                                       product_p[i] + self.Get_erf_LB(product_quality[i], [lb[13 + i], ub[13 + i]],
                                                                      [lb[13 + i], ub[13 + i]])[0],
                                       product_p[i] + (first_intersect + first_slope * (product_mus[i] - lb[13 + i]))
                                       )

                    lbg.extend([-np.inf, -np.inf])
                    ubg.extend([0, 0])

            original_variables = \
                casadi.vertcat(feed_flows, pool_flows, product_flows, pool_mus, pool_sigmas, product_mus,
                               product_sigmas,
                               product_p)
            num_original_variables = original_variables.shape[0]

            all_variables = casadi.vertcat(original_variables, feed_flows_squared, pool_mu_bilinears,
                                           pool_sigma_bilinears,
                                           pool_sigma_squared_bilinears, product_mu_bilinears, product_sigma_bilinears,
                                           product_sigma_squared_bilinears, pool_sigma_bilinears_sums,
                                           pool_sigma_bilinears_sums_squared)

            all_variables_names = [all_variables[i].name() for i in range(all_variables.shape[0])]

            # Add all the bilinear relationships to a list such that the convex relaxations can be added to the problem
            # formulation

            bilinear_terms = [[feed_flows[i], feed_flows[i], feed_flows_squared[i]] for i in range(3)]

            bilinear_terms.extend(
                [[pool_mus[i], pool_flows[j], pool_mu_bilinears[j]] for i in range(2) for j in
                 range(2 * i, 2 * (i + 1))])

            bilinear_terms.extend([[pool_sigmas[i], pool_flows[j], pool_sigma_bilinears[j]] for i in range(2) for j in
                                   range(2 * i, 2 * (i + 1))])

            bilinear_terms.extend(
                [[pool_sigma_bilinears[i], pool_sigma_bilinears[i], pool_sigma_squared_bilinears[i]] for i in
                 range(4)])

            bilinear_terms.extend([[product_mus[i], product_flows[i], product_mu_bilinears[i]] for i in range(2)])

            bilinear_terms.extend([[product_sigmas[i], product_flows[i], product_sigma_bilinears[i]] for i in range(2)])

            bilinear_terms.extend(
                [[product_sigma_bilinears[i], product_sigma_bilinears[i], product_sigma_squared_bilinears[i]] for i in
                 range(2)])

            bilinear_terms.extend(
                [[pool_sigma_bilinears_sums[i], pool_sigma_bilinears_sums[i], pool_sigma_bilinears_sums_squared[i]] for
                 i in
                 range(2)])

            summed_terms = [
                [pool_sigma_bilinears[i * 2], pool_sigma_bilinears[i * 2 + 1],
                 pool_sigma_bilinears_sums[i]] for i in range(2)]

            # We create mirror structures of the ones containing the bilinear terms where instead of the term we focus
            # on the index the variables have in the list of all variables. This is so we can easily access the right
            # variables later.

            bilinear_names = []
            bilinear_indices = []

            for i in range(len(bilinear_terms)):
                bilinear_names.append(
                    [bilinear_terms[i][0].name(), bilinear_terms[i][1].name(), bilinear_terms[i][2].name()])
                bilinear_indices.append([all_variables_names.index(bilinear_terms[i][0].name()),
                                         all_variables_names.index(bilinear_terms[i][1].name()),
                                         all_variables_names.index(bilinear_terms[i][2].name())])

            summed_names = []
            summed_indices = []

            for i in range(len(summed_terms)):
                summed_names.append([summed_terms[i][j].name() for j in range(len(summed_terms[i]))])
                summed_indices.append(
                    [all_variables_names.index(summed_terms[i][j].name()) for j in range(len(summed_terms[i]))])

            lb.extend([np.nan for i in range(all_variables.shape[0] - original_variables.shape[0])])
            ub.extend([np.nan for i in range(all_variables.shape[0] - original_variables.shape[0])])

            for i in range(len(bilinear_indices)):
                lb[bilinear_indices[i][2]] = lb[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]]
                ub[bilinear_indices[i][2]] = ub[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]

            for i in range(len(summed_indices)):
                lb[summed_indices[i][-1]] = sum([lb[summed_indices[i][j]] for j in range(len(summed_indices[i]) - 1)])
                ub[summed_indices[i][-1]] = sum([ub[summed_indices[i][j]] for j in range(len(summed_indices[i]) - 1)])

            for i in range(len(bilinear_indices)):
                lb[bilinear_indices[i][2]] = lb[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]]
                ub[bilinear_indices[i][2]] = ub[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]

            # Finally we can add the actual McCormick relaxation terms to the constraint set
            for i in range(len(bilinear_indices)):
                g = casadi.vertcat(g,
                                   -all_variables[bilinear_indices[i][2]] + lb[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] +
                                   all_variables[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]] - lb[
                                       bilinear_indices[i][0]] * lb[bilinear_indices[i][1]],
                                   -all_variables[bilinear_indices[i][2]] + ub[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] +
                                   all_variables[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]] - ub[
                                       bilinear_indices[i][0]] * ub[bilinear_indices[i][1]],
                                   all_variables[bilinear_indices[i][2]] - ub[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] -
                                   all_variables[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]] + ub[
                                       bilinear_indices[i][0]] * lb[bilinear_indices[i][1]],
                                   all_variables[bilinear_indices[i][2]] - all_variables[bilinear_indices[i][0]] * ub[
                                       bilinear_indices[i][1]] -
                                   lb[bilinear_indices[i][0]] * all_variables[bilinear_indices[i][1]] + lb[
                                       bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]
                                   )
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                ubg.append(0)
                ubg.append(0)
                ubg.append(0)
                ubg.append(0)
        elif self.problem_name == 'Haverly_3':
            # Set up the casadi variables needed for the problem
            feed_flows = casadi.SX.sym('feed_flows', 3)
            pool_flows = casadi.SX.sym('pool_flows', 4)
            product_flows = casadi.SX.sym('product_flows', 2)
            pool_mus = casadi.SX.sym('pool_mus', 2)
            pool_sigmas = casadi.SX.sym('pool_sigmas', 2)
            product_mus = casadi.SX.sym('product_mus', 2)
            product_sigmas = casadi.SX.sym('product_sigma', 2)
            product_p = casadi.SX.sym('product_probabilities', 2)

            # Specify the problem parameters
            feed_price = [6, 13, 10]
            feed_mus = [3, 1, 2]

            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            # self.update_bounds(node)

            lb = list(self.lower_bounds[node].copy())
            ub = list(self.upper_bounds[node].copy())

            # Set the objective function of the problem
            objective = casadi.sum1(feed_flows * feed_price) - casadi.sum1(product_p * product_flows * product_price)

            # Initiate a casadi SX object, and two lists to gradually build the constraints of the problem
            g = casadi.SX()
            lbg = []
            ubg = []

            # Pool mass balance
            g = casadi.vertcat(g,
                               feed_flows[0] + feed_flows[1] - casadi.sum1(pool_flows[:2]),
                               feed_flows[2] - casadi.sum1(pool_flows[2:4]),
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mass balance

            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 2 for i in range(2)]]) - product_flows[0],
                               casadi.sum1(pool_flows[[i * 2 + 1 for i in range(2)]]) - product_flows[1],
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool mu balance

            pool_mu_bilinears = casadi.SX.sym('pool_mu_bilinears', 4)

            g = casadi.vertcat(g,
                               feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1] - casadi.sum1(
                                   pool_mu_bilinears[:2]),
                               feed_flows[2] * feed_mus[2] - casadi.sum1(pool_mu_bilinears[2:4])
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Next we have to define a number of artificial problem variables in order to obtain bilinear terms
            # that we can more easily define the convex relaxations for.
            pool_sigma_bilinears = casadi.SX.sym('pool_sigma_bilinears', 4)
            pool_sigma_squared_bilinears = casadi.SX.sym('pool_sigma_squared_bilinears', 4)
            pool_sigma_bilinears_sums = casadi.SX.sym('pool_sigma_bilinears_sums', 2)
            pool_sigma_bilinears_sums_squared = casadi.SX.sym('pool_sigma_bilinears_sums_squared', 2)

            feed_flows_squared = casadi.SX.sym('feed_flows_squared', 3)

            # Pool sigma balance
            g = casadi.vertcat(g,
                               pool_sigma_bilinears_sums[0] - casadi.sum1(pool_sigma_bilinears[:2]),
                               pool_sigma_bilinears_sums[1] - casadi.sum1(pool_sigma_bilinears[2:4])
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            g = casadi.vertcat(g,
                               feed_flows_squared[0] * self.std ** 2 + feed_flows_squared[1] * self.std ** 2 -
                               pool_sigma_bilinears_sums_squared[0],
                               feed_flows_squared[2] * self.std ** 2 - pool_sigma_bilinears_sums_squared[1],
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mu balance

            product_mu_bilinears = casadi.SX.sym('product_mu_bilinears', 2)

            g = casadi.vertcat(g,
                               casadi.sum1(pool_mu_bilinears[[i * 2 for i in range(2)]]) - product_mu_bilinears[0],
                               casadi.sum1(pool_mu_bilinears[[i * 2 + 1 for i in range(2)]]) - product_mu_bilinears[1]
                               )
            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product sigma balance

            product_sigma_bilinears = casadi.SX.sym('product_sigma_bilienars', 2)
            product_sigma_squared_bilinears = casadi.SX.sym('product_sigma_squared_bilinears', 2)

            g = casadi.vertcat(g,
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 2 for i in range(2)]]) -
                               product_sigma_squared_bilinears[0],
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 2 + 1 for i in range(2)]]) -
                               product_sigma_squared_bilinears[1]
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Underestimating the cumulative distribution function
            for i in range(2):
                if ub[13 + i] <= product_quality[i]:
                    g = casadi.vertcat(g,
                                       product_p[i] - 0.5 * (1 + casadi.erf(
                                           (product_quality[i] - product_mus[i]) / (product_sigmas[i] * np.sqrt(2))))
                                       )
                    lbg.append(-np.inf)
                    ubg.append(0)
                else:
                    first_intersect, first_slope = \
                        self.Get_erf_LB(product_quality[i], [lb[13 + i], ub[13 + i]], [lb[13 + i], ub[13 + i]])[1]

                    g = casadi.vertcat(g,
                                       product_p[i] + self.Get_erf_LB(product_quality[i], [lb[13 + i], ub[13 + i]],
                                                                      [lb[13 + i], ub[13 + i]])[0],
                                       product_p[i] + (first_intersect + first_slope * (product_mus[i] - lb[13 + i]))
                                       )

                    lbg.extend([-np.inf, -np.inf])
                    ubg.extend([0, 0])

            original_variables = \
                casadi.vertcat(feed_flows, pool_flows, product_flows, pool_mus, pool_sigmas, product_mus,
                               product_sigmas,
                               product_p)
            num_original_variables = original_variables.shape[0]

            all_variables = casadi.vertcat(original_variables, feed_flows_squared, pool_mu_bilinears,
                                           pool_sigma_bilinears,
                                           pool_sigma_squared_bilinears, product_mu_bilinears, product_sigma_bilinears,
                                           product_sigma_squared_bilinears, pool_sigma_bilinears_sums,
                                           pool_sigma_bilinears_sums_squared)

            all_variables_names = [all_variables[i].name() for i in range(all_variables.shape[0])]

            # Add all the bilinear relationships to a list such that the convex relaxations can be added to the problem
            # formulation.
            bilinear_terms = [[feed_flows[i], feed_flows[i], feed_flows_squared[i]] for i in range(3)]

            bilinear_terms.extend(
                [[pool_mus[i], pool_flows[j], pool_mu_bilinears[j]] for i in range(2) for j in
                 range(2 * i, 2 * (i + 1))])

            bilinear_terms.extend([[pool_sigmas[i], pool_flows[j], pool_sigma_bilinears[j]] for i in range(2) for j in
                                   range(2 * i, 2 * (i + 1))])

            bilinear_terms.extend(
                [[pool_sigma_bilinears[i], pool_sigma_bilinears[i], pool_sigma_squared_bilinears[i]] for i in
                 range(4)])

            bilinear_terms.extend([[product_mus[i], product_flows[i], product_mu_bilinears[i]] for i in range(2)])

            bilinear_terms.extend([[product_sigmas[i], product_flows[i], product_sigma_bilinears[i]] for i in range(2)])

            bilinear_terms.extend(
                [[product_sigma_bilinears[i], product_sigma_bilinears[i], product_sigma_squared_bilinears[i]] for i in
                 range(2)])

            bilinear_terms.extend(
                [[pool_sigma_bilinears_sums[i], pool_sigma_bilinears_sums[i], pool_sigma_bilinears_sums_squared[i]] for
                 i in
                 range(2)])

            summed_terms = [
                [pool_sigma_bilinears[i * 2], pool_sigma_bilinears[i * 2 + 1],
                 pool_sigma_bilinears_sums[i]] for i in range(2)]

            # We create mirror structures of the ones containing the bilinear terms where instead of the term we focus
            # on the index the variables have in the list of all variables. This is so we can easily access the right
            # variables later.
            bilinear_names = []
            bilinear_indices = []

            for i in range(len(bilinear_terms)):
                bilinear_names.append(
                    [bilinear_terms[i][0].name(), bilinear_terms[i][1].name(), bilinear_terms[i][2].name()])
                bilinear_indices.append([all_variables_names.index(bilinear_terms[i][0].name()),
                                         all_variables_names.index(bilinear_terms[i][1].name()),
                                         all_variables_names.index(bilinear_terms[i][2].name())])

            summed_names = []
            summed_indices = []

            for i in range(len(summed_terms)):
                summed_names.append([summed_terms[i][j].name() for j in range(len(summed_terms[i]))])
                summed_indices.append(
                    [all_variables_names.index(summed_terms[i][j].name()) for j in range(len(summed_terms[i]))])

            lb.extend([np.nan for i in range(all_variables.shape[0] - original_variables.shape[0])])
            ub.extend([np.nan for i in range(all_variables.shape[0] - original_variables.shape[0])])

            for i in range(len(bilinear_indices)):
                lb[bilinear_indices[i][2]] = lb[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]]
                ub[bilinear_indices[i][2]] = ub[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]

            for i in range(len(summed_indices)):
                lb[summed_indices[i][-1]] = sum([lb[summed_indices[i][j]] for j in range(len(summed_indices[i]) - 1)])
                ub[summed_indices[i][-1]] = sum([ub[summed_indices[i][j]] for j in range(len(summed_indices[i]) - 1)])

            for i in range(len(bilinear_indices)):
                lb[bilinear_indices[i][2]] = lb[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]]
                ub[bilinear_indices[i][2]] = ub[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]

            # Finally we can add the actual McCormick relaxation terms to the constraint set
            for i in range(len(bilinear_indices)):
                g = casadi.vertcat(g,
                                   -all_variables[bilinear_indices[i][2]] + lb[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] +
                                   all_variables[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]] - lb[
                                       bilinear_indices[i][0]] * lb[bilinear_indices[i][1]],
                                   -all_variables[bilinear_indices[i][2]] + ub[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] +
                                   all_variables[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]] - ub[
                                       bilinear_indices[i][0]] * ub[bilinear_indices[i][1]],
                                   all_variables[bilinear_indices[i][2]] - ub[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] -
                                   all_variables[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]] + ub[
                                       bilinear_indices[i][0]] * lb[bilinear_indices[i][1]],
                                   all_variables[bilinear_indices[i][2]] - all_variables[bilinear_indices[i][0]] * ub[
                                       bilinear_indices[i][1]] -
                                   lb[bilinear_indices[i][0]] * all_variables[bilinear_indices[i][1]] + lb[
                                       bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]
                                   )
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                ubg.append(0)
                ubg.append(0)
                ubg.append(0)
                ubg.append(0)
        elif self.problem_name == 'Foulds_2':
            # Set up the casadi variables needed for the problem
            feed_flows = casadi.SX.sym('feed_flows', 6)
            pool_flows = casadi.SX.sym('pool_flows', 16)
            product_flows = casadi.SX.sym('product_flows', 4)
            pool_mus = casadi.SX.sym('pool_mus', 4)
            pool_sigmas = casadi.SX.sym('pool_sigmas', 4)
            product_mus = casadi.SX.sym('product_mus', 4)
            product_sigmas = casadi.SX.sym('product_sigma', 4)
            product_p = casadi.SX.sym('product_probabilities', 4)

            # Specify the problem parameters
            feed_price = [6, 16, 10, 3, 13, 7]
            feed_mus = [3, 1, 2, 3.5, 1.5, 2.5]

            product_price = [9, 15, 6, 12]
            product_quality = [2.5, 1.5, 3, 2]

            #self.update_bounds(node)

            lb = list(self.lower_bounds[node].copy())
            ub = list(self.upper_bounds[node].copy())

            # Set the objective function of the problem
            objective = casadi.sum1(feed_flows * feed_price) - casadi.sum1(product_p * product_flows * product_price)

            # Initiate a casadi SX object, and two lists to gradually build the constraints of the problem
            g = casadi.SX()
            lbg = []
            ubg = []

            # Pool mass balance

            g = casadi.vertcat(g,
                               feed_flows[0] + feed_flows[1] - casadi.sum1(pool_flows[:4]),
                               feed_flows[2] - casadi.sum1(pool_flows[4:8]),
                               feed_flows[3] + feed_flows[4] - casadi.sum1(pool_flows[8:12]),
                               feed_flows[5] - casadi.sum1(pool_flows[12:16])
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Product mass balance

            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 4 for i in range(4)]]) - product_flows[0],
                               casadi.sum1(pool_flows[[i * 4 + 1 for i in range(4)]]) - product_flows[1],
                               casadi.sum1(pool_flows[[i * 4 + 2 for i in range(4)]]) - product_flows[2],
                               casadi.sum1(pool_flows[[i * 4 + 3 for i in range(4)]]) - product_flows[3]
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Pool mu balance

            pool_mu_bilinears = casadi.SX.sym('pool_mu_bilinears', 16)

            g = casadi.vertcat(g,
                               feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1] - casadi.sum1(
                                   pool_mu_bilinears[:4]),
                               feed_flows[2] * feed_mus[2] - casadi.sum1(pool_mu_bilinears[4:8]),
                               feed_flows[3] * feed_mus[3] + feed_flows[4] * feed_mus[4] - casadi.sum1(
                                   pool_mu_bilinears[8:12]),
                               feed_flows[5] * feed_mus[5] - casadi.sum1(pool_mu_bilinears[12:16])
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Next we have to define a number of artificial problem variables in order to obtain bilinear terms
            # that we can more easily define the convex relaxations for.
            pool_sigma_bilinears = casadi.SX.sym('pool_sigma_bilinears', 16)
            pool_sigma_squared_bilinears = casadi.SX.sym('pool_sigma_squared_bilinears', 16)
            pool_sigma_bilinears_sums = casadi.SX.sym('pool_sigma_bilinears_sums', 4)
            pool_sigma_bilinears_sums_squared = casadi.SX.sym('pool_sigma_bilinears_sums_squared', 4)

            feed_flows_squared = casadi.SX.sym('feed_flows_squared', 6)

            # Pool sigma balance
            g = casadi.vertcat(g,
                               pool_sigma_bilinears_sums[0] - casadi.sum1(pool_sigma_bilinears[:4]),
                               pool_sigma_bilinears_sums[1] - casadi.sum1(pool_sigma_bilinears[4:8]),
                               pool_sigma_bilinears_sums[2] - casadi.sum1(pool_sigma_bilinears[8:12]),
                               pool_sigma_bilinears_sums[3] - casadi.sum1(pool_sigma_bilinears[12:16])
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            g = casadi.vertcat(g,
                               feed_flows_squared[0] * self.std ** 2 + feed_flows_squared[1] * self.std ** 2 -
                               pool_sigma_bilinears_sums_squared[0],
                               feed_flows_squared[2] * self.std ** 2 - pool_sigma_bilinears_sums_squared[1],
                               feed_flows_squared[3] * self.std ** 2 + feed_flows_squared[4] * self.std ** 2 -
                               pool_sigma_bilinears_sums_squared[2],
                               feed_flows_squared[5] * self.std ** 2 - pool_sigma_bilinears_sums_squared[3]
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Product mu balance

            product_mu_bilinears = casadi.SX.sym('product_mu_bilinears', 4)

            g = casadi.vertcat(g,
                               casadi.sum1(pool_mu_bilinears[[i * 4 for i in range(4)]]) - product_mu_bilinears[0],
                               casadi.sum1(pool_mu_bilinears[[i * 4 + 1 for i in range(4)]]) - product_mu_bilinears[1],
                               casadi.sum1(pool_mu_bilinears[[i * 4 + 2 for i in range(4)]]) - product_mu_bilinears[2],
                               casadi.sum1(pool_mu_bilinears[[i * 4 + 3 for i in range(4)]]) - product_mu_bilinears[3]
                               )
            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Product sigma balance

            product_sigma_bilinears = casadi.SX.sym('product_sigma_bilienars', 4)
            product_sigma_squared_bilinears = casadi.SX.sym('product_sigma_squared_bilinears', 4)

            g = casadi.vertcat(g,
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 4 for i in range(4)]]) -
                               product_sigma_squared_bilinears[0],
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 4 + 1 for i in range(4)]]) -
                               product_sigma_squared_bilinears[1],
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 4 + 2 for i in range(4)]]) -
                               product_sigma_squared_bilinears[2],
                               casadi.sum1(pool_sigma_squared_bilinears[[i * 4 + 3 for i in range(4)]]) -
                               product_sigma_squared_bilinears[3]
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Convex relaxation of the cumulative distribution function
            for i in range(4):
                print(i)
                if ub[34 + i] <= product_quality[i]:
                    g = casadi.vertcat(g,
                                       product_p[i] - 0.5 * (1 + casadi.erf(
                                           (product_quality[i] - product_mus[i]) / (product_sigmas[i] * np.sqrt(2))))
                                       )
                    lbg.append(-np.inf)
                    ubg.append(0)
                else:
                    first_intersect, first_slope = \
                    self.Get_erf_LB(product_quality[i], [lb[34 + i], ub[34 + i]], [lb[38 + i], ub[38 + i]])[1]

                    g = casadi.vertcat(g,
                                       product_p[i] + self.Get_erf_LB(product_quality[i], [lb[34 + i], ub[34 + i]],
                                                                      [lb[38 + i], ub[38 + i]])[0],
                                       product_p[i] + (first_intersect + first_slope * (product_mus[i] - lb[34 + i]))
                                       )

                    lbg.extend([-np.inf, -np.inf])
                    ubg.extend([0, 0])

            original_variables = \
                casadi.vertcat(feed_flows, pool_flows, product_flows, pool_mus, pool_sigmas, product_mus, product_sigmas,
                               product_p)
            num_original_variables = original_variables.shape[0]

            all_variables = casadi.vertcat(original_variables, feed_flows_squared, pool_mu_bilinears, pool_sigma_bilinears,
                                           pool_sigma_squared_bilinears, product_mu_bilinears, product_sigma_bilinears,
                                           product_sigma_squared_bilinears, pool_sigma_bilinears_sums,
                                           pool_sigma_bilinears_sums_squared)

            all_variables_names = [all_variables[i].name() for i in range(all_variables.shape[0])]

            # Add all the bilinear relationships to a list such that the convex relaxations can be added to the problem
            # formulation.
            bilinear_terms = [[feed_flows[i], feed_flows[i], feed_flows_squared[i]] for i in range(6)]

            bilinear_terms.extend(
                [[pool_mus[i], pool_flows[j], pool_mu_bilinears[j]] for i in range(4) for j in range(4 * i, 4 * (i + 1))])

            bilinear_terms.extend([[pool_sigmas[i], pool_flows[j], pool_sigma_bilinears[j]] for i in range(4) for j in
                                   range(4 * i, 4 * (i + 1))])

            bilinear_terms.extend(
                [[pool_sigma_bilinears[i], pool_sigma_bilinears[i], pool_sigma_squared_bilinears[i]] for i in range(16)])

            bilinear_terms.extend([[product_mus[i], product_flows[i], product_mu_bilinears[i]] for i in range(4)])

            bilinear_terms.extend([[product_sigmas[i], product_flows[i], product_sigma_bilinears[i]] for i in range(4)])

            bilinear_terms.extend(
                [[product_sigma_bilinears[i], product_sigma_bilinears[i], product_sigma_squared_bilinears[i]] for i in
                 range(4)])

            bilinear_terms.extend(
                [[pool_sigma_bilinears_sums[i], pool_sigma_bilinears_sums[i], pool_sigma_bilinears_sums_squared[i]] for i in
                 range(4)])

            summed_terms = [[pool_sigma_bilinears[i * 4], pool_sigma_bilinears[i * 4 + 1], pool_sigma_bilinears[i * 4 + 2],
                             pool_sigma_bilinears[i * 4 + 3], pool_sigma_bilinears_sums[i]] for i in range(4)]

            # We create mirror structures of the ones containing the bilinear terms where instead of the term we focus
            # on the index the variables have in the list of all variables. This is so we can easily access the right
            # variables later.
            bilinear_names = []
            bilinear_indices = []

            for i in range(len(bilinear_terms)):
                bilinear_names.append(
                    [bilinear_terms[i][0].name(), bilinear_terms[i][1].name(), bilinear_terms[i][2].name()])
                bilinear_indices.append([all_variables_names.index(bilinear_terms[i][0].name()),
                                         all_variables_names.index(bilinear_terms[i][1].name()),
                                         all_variables_names.index(bilinear_terms[i][2].name())])

            summed_names = []
            summed_indices = []

            for i in range(len(summed_terms)):
                summed_names.append([summed_terms[i][j].name() for j in range(len(summed_terms[i]))])
                summed_indices.append(
                    [all_variables_names.index(summed_terms[i][j].name()) for j in range(len(summed_terms[i]))])

            lb.extend([np.nan for i in range(all_variables.shape[0] - original_variables.shape[0])])
            ub.extend([np.nan for i in range(all_variables.shape[0] - original_variables.shape[0])])

            for i in range(len(bilinear_indices)):
                lb[bilinear_indices[i][2]] = lb[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]]
                ub[bilinear_indices[i][2]] = ub[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]

            for i in range(len(summed_indices)):
                lb[summed_indices[i][-1]] = sum([lb[summed_indices[i][j]] for j in range(len(summed_indices[i]) - 1)])
                ub[summed_indices[i][-1]] = sum([ub[summed_indices[i][j]] for j in range(len(summed_indices[i]) - 1)])

            for i in range(len(bilinear_indices)):
                lb[bilinear_indices[i][2]] = lb[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]]
                ub[bilinear_indices[i][2]] = ub[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]

            # Finally we can add the actual McCormick relaxation terms to the constraint set
            for i in range(len(bilinear_indices)):
                g = casadi.vertcat(g,
                                   -all_variables[bilinear_indices[i][2]] + lb[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] +
                                   all_variables[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]] - lb[
                                       bilinear_indices[i][0]] * lb[bilinear_indices[i][1]],
                                   -all_variables[bilinear_indices[i][2]] + ub[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] +
                                   all_variables[bilinear_indices[i][0]] * ub[bilinear_indices[i][1]] - ub[
                                       bilinear_indices[i][0]] * ub[bilinear_indices[i][1]],
                                   all_variables[bilinear_indices[i][2]] - ub[bilinear_indices[i][0]] * all_variables[
                                       bilinear_indices[i][1]] -
                                   all_variables[bilinear_indices[i][0]] * lb[bilinear_indices[i][1]] + ub[
                                       bilinear_indices[i][0]] * lb[bilinear_indices[i][1]],
                                   all_variables[bilinear_indices[i][2]] - all_variables[bilinear_indices[i][0]] * ub[
                                       bilinear_indices[i][1]] -
                                   lb[bilinear_indices[i][0]] * all_variables[bilinear_indices[i][1]] + lb[
                                       bilinear_indices[i][0]] * ub[bilinear_indices[i][1]]
                                   )
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                lbg.append(-np.inf)
                ubg.append(0)
                ubg.append(0)
                ubg.append(0)
                ubg.append(0)
        elif self.problem_name == 'Segarwak':
            pass

        # Now the convex relaxation of the original problem with the given problem bounds can be solved using IPOPT
        # through CasADI.
        problem = {'x': all_variables, 'f': objective, 'g': g}
        solver = casadi.nlpsol('solver', 'ipopt', problem)
        solution = solver(lbx=lb, ubx=ub, lbg=lbg, ubg=ubg)

        if solver.stats()['success'] is True:
            print('This is a lower bound')
            print(solution['f'])
            return [solver.stats()['success'], float(solution['f'])]
        else:
            print('This is a lower bound')
            print(np.inf)
            return [False, np.inf]

    def upper_bounding(self, node):
        """
        Method to solve an upper bounding problem for the stochastic programming approach. This locally optimises the
        nonconvex problem to obtain a feasible solution that can eb used for spatial branch and bound pruning.
        """
        if self.problem_name == 'Haverly_1' or self.problem_name == 'Haverly_2':
            # Defining the casadi variables needed to for the original problem
            feed_flows = casadi.SX.sym('feed_flows', 3)
            pool_flows = casadi.SX.sym('pool_flows', 4)
            product_flows = casadi.SX.sym('product_flows', 2)
            pool_mus = casadi.SX.sym('pool_mus', 2)
            pool_sigmas = casadi.SX.sym('pool_sigmas', 2)
            product_mus = casadi.SX.sym('product_mus', 2)
            product_sigmas = casadi.SX.sym('product_sigma', 2)
            product_p = casadi.SX.sym('product_probabilities', 2)

            all_variables = \
                casadi.vertcat(feed_flows, pool_flows, product_flows, pool_mus, pool_sigmas, product_mus,
                               product_sigmas,
                               product_p)

            # Specifying the problem parameters
            feed_price = [6, 16, 10]
            feed_mus = [3, 1, 2]

            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            lb = list(self.lower_bounds[node].copy())
            ub = list(self.upper_bounds[node].copy())

            # Setting the objective function for the problem
            objective = casadi.sum1(feed_flows * feed_price) - casadi.sum1(product_p * product_flows * product_price)

            # Initialising the computational structures used to represent the constraint set of the problem
            g = casadi.SX()
            lbg = []
            ubg = []

            # Pool mass balance
            g = casadi.vertcat(g,
                               feed_flows[0] + feed_flows[1] - casadi.sum1(pool_flows[:2]),
                               feed_flows[2] - casadi.sum1(pool_flows[2:4])
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mass balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 2 for i in range(2)]]) - product_flows[0],
                               casadi.sum1(pool_flows[[i * 2 + 1 for i in range(2)]]) - product_flows[1]
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool mu balance
            g = casadi.vertcat(g,
                               feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1] - casadi.sum1(
                                   pool_flows[:2]) * pool_mus[0],
                               feed_flows[2] * feed_mus[2] - casadi.sum1(pool_flows[2:3]) * pool_mus[1]
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool sigma balance
            g = casadi.vertcat(g,
                               feed_flows[0] ** 2 * self.std ** 2 + feed_flows[1] ** 2 * self.std ** 2 -
                               (casadi.sum1(pool_flows[:2]) ** 2 * pool_sigmas[0] ** 2),
                               feed_flows[2] ** 2 * self.std ** 2 - (
                                       casadi.sum1(pool_flows[2:4]) ** 2 * pool_sigmas[1] ** 2)
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mu balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 2 for i in range(2)]] * pool_mus[[i for i in range(2)]]) -
                               product_flows[0] * product_mus[0],
                               casadi.sum1(pool_flows[[i * 2 + 1 for i in range(2)]] * pool_mus[[i for i in range(2)]])
                               - product_flows[1] * product_mus[1]
                               )
            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product sigma balance
            g = casadi.vertcat(g,
                               casadi.sum1(
                                   pool_flows[[i * 2 for i in range(2)]] ** 2 * pool_sigmas[
                                       [i for i in range(2)]] ** 2) -
                               product_flows[0] ** 2 * product_sigmas[0] ** 2,
                               casadi.sum1(
                                   pool_flows[[i * 2 + 1 for i in range(2)]] ** 2 * pool_sigmas[
                                       [i for i in range(2)]] ** 2)
                               - product_flows[1] ** 2 * product_sigmas[1] ** 2
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Deriving the quality satisfaction probability from the cumulative distribution function
            for i in range(2):
                g = casadi.vertcat(g,
                                   0.5 * (1 + casadi.erf(
                                       (product_quality[i] - product_mus[i]) / (product_sigmas[i] * np.sqrt(2)))) -
                                   product_p[i]
                                   )
                lbg.append(0)
                ubg.append(np.inf)
                # ubg.append(0) ## Technically this should be a an equality constraint. In practice, the objective is monotonic in the probabilities so writing it as an inequality is valid.
        elif self.problem_name == 'Haverly_3':
            # Defining the casadi variables needed to for the original problem
            feed_flows = casadi.SX.sym('feed_flows', 3)
            pool_flows = casadi.SX.sym('pool_flows', 4)
            product_flows = casadi.SX.sym('product_flows', 2)
            pool_mus = casadi.SX.sym('pool_mus', 2)
            pool_sigmas = casadi.SX.sym('pool_sigmas', 2)
            product_mus = casadi.SX.sym('product_mus', 2)
            product_sigmas = casadi.SX.sym('product_sigma', 2)
            product_p = casadi.SX.sym('product_probabilities', 2)

            all_variables = \
                casadi.vertcat(feed_flows, pool_flows, product_flows, pool_mus, pool_sigmas, product_mus,
                               product_sigmas,
                               product_p)

            # Specifying the problem parameters
            feed_price = [6, 13, 10]
            feed_mus = [3, 1, 2]

            product_price = [9, 15]
            product_quality = [2.5, 1.5]

            lb = list(self.lower_bounds[node].copy())
            ub = list(self.upper_bounds[node].copy())

            # Setting the objective function of the problem
            objective = casadi.sum1(feed_flows * feed_price) - casadi.sum1(product_p * product_flows * product_price)

            # Initialising the computational structures used to represent the constraint set of the problem
            g = casadi.SX()
            lbg = []
            ubg = []

            # Pool mass balance
            g = casadi.vertcat(g,
                               feed_flows[0] + feed_flows[1] - casadi.sum1(pool_flows[:2]),
                               feed_flows[2] - casadi.sum1(pool_flows[2:4])
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mass balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 2 for i in range(2)]]) - product_flows[0],
                               casadi.sum1(pool_flows[[i * 2 + 1 for i in range(2)]]) - product_flows[1]
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool mu balance
            g = casadi.vertcat(g,
                               feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1] - casadi.sum1(
                                   pool_flows[:2]) * pool_mus[0],
                               feed_flows[2] * feed_mus[2] - casadi.sum1(pool_flows[2:3]) * pool_mus[1]
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Pool sigma balance
            g = casadi.vertcat(g,
                               feed_flows[0] ** 2 * self.std ** 2 + feed_flows[1] ** 2 * self.std ** 2 -
                               (casadi.sum1(pool_flows[:2]) ** 2 * pool_sigmas[0] ** 2),
                               feed_flows[2] ** 2 * self.std ** 2 - (
                                       casadi.sum1(pool_flows[2:4]) ** 2 * pool_sigmas[1] ** 2)
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product mu balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 2 for i in range(2)]] * pool_mus[[i for i in range(2)]]) -
                               product_flows[0] * product_mus[0],
                               casadi.sum1(pool_flows[[i * 2 + 1 for i in range(2)]] * pool_mus[[i for i in range(2)]])
                               - product_flows[1] * product_mus[1]
                               )
            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Product sigma balance
            g = casadi.vertcat(g,
                               casadi.sum1(
                                   pool_flows[[i * 2 for i in range(2)]] ** 2 * pool_sigmas[
                                       [i for i in range(2)]] ** 2) -
                               product_flows[0] ** 2 * product_sigmas[0] ** 2,
                               casadi.sum1(
                                   pool_flows[[i * 2 + 1 for i in range(2)]] ** 2 * pool_sigmas[
                                       [i for i in range(2)]] ** 2)
                               - product_flows[1] ** 2 * product_sigmas[1] ** 2
                               )

            lbg.extend([0, 0])
            ubg.extend([0, 0])

            # Deriving the quality satisfaction probability from the cumulative distribution function
            for i in range(2):
                g = casadi.vertcat(g,
                                   0.5 * (1 + casadi.erf(
                                       (product_quality[i] - product_mus[i]) / (product_sigmas[i] * np.sqrt(2)))) -
                                   product_p[i]
                                   )
                lbg.append(0)
                ubg.append(np.inf)
                # ubg.append(0) ## Technically this should be a an equality constraint. In practice, the objective is monotonic in the probabilities so writing it as an inequality is valid.
        elif self.problem_name == 'Foulds_2':
            # Defining the casadi variables needed to for the original problem
            feed_flows = casadi.SX.sym('feed_flows', 6)
            pool_flows = casadi.SX.sym('pool_flows', 16)
            product_flows = casadi.SX.sym('product_flows', 4)
            pool_mus = casadi.SX.sym('pool_mus', 4)
            pool_sigmas = casadi.SX.sym('pool_sigmas', 4)
            product_mus = casadi.SX.sym('product_mus', 4)
            product_sigmas = casadi.SX.sym('product_sigma', 4)
            product_p = casadi.SX.sym('product_probabilities', 4)

            all_variables = \
                casadi.vertcat(feed_flows, pool_flows, product_flows, pool_mus, pool_sigmas, product_mus,
                               product_sigmas,
                               product_p)

            # Specifying the problem parameters
            feed_price = [6, 16, 10, 3, 13, 7]
            feed_mus = [3, 1, 2, 3.5, 1.5, 2.5]

            product_price = [9, 15, 6, 12]
            product_quality = [2.5, 1.5, 3, 2]

            lb = list(self.lower_bounds[node].copy())
            ub = list(self.upper_bounds[node].copy())

            # Setting the objective function of the problem
            objective = casadi.sum1(feed_flows * feed_price) - casadi.sum1(product_p * product_flows * product_price)

            # Initialising the computational structures used to represent the constraint set of the problem
            g = casadi.SX()
            lbg = []
            ubg = []

            # Pool mass balance
            g = casadi.vertcat(g,
                               feed_flows[0] + feed_flows[1] - casadi.sum1(pool_flows[:4]),
                               feed_flows[2] - casadi.sum1(pool_flows[4:8]),
                               feed_flows[3] + feed_flows[4] - casadi.sum1(pool_flows[8:12]),
                               feed_flows[5] - casadi.sum1(pool_flows[12:16])
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Product mass balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 4 for i in range(4)]]) - product_flows[0],
                               casadi.sum1(pool_flows[[i * 4 + 1 for i in range(4)]]) - product_flows[1],
                               casadi.sum1(pool_flows[[i * 4 + 2 for i in range(4)]]) - product_flows[2],
                               casadi.sum1(pool_flows[[i * 4 + 3 for i in range(4)]]) - product_flows[3]
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Pool mu balance
            g = casadi.vertcat(g,
                               feed_flows[0] * feed_mus[0] + feed_flows[1] * feed_mus[1] - casadi.sum1(
                                   pool_flows[:4]) * pool_mus[0],
                               feed_flows[2] * feed_mus[2] - casadi.sum1(pool_flows[4:8]) * pool_mus[1],
                               feed_flows[3] * feed_mus[3] + feed_flows[4] * feed_mus[4] - casadi.sum1(
                                   pool_flows[8:12]) * pool_mus[2],
                               feed_flows[5] * feed_mus[5] - casadi.sum1(pool_flows[12:16]) * pool_mus[3]
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Pool sigma balance
            g = casadi.vertcat(g,
                               feed_flows[0] ** 2 * self.std ** 2 + feed_flows[1] ** 2 * self.std ** 2 -
                               (casadi.sum1(pool_flows[:4]) ** 2 * pool_sigmas[0] ** 2),
                               feed_flows[2] ** 2 * self.std ** 2 - (
                                       casadi.sum1(pool_flows[4:8]) ** 2 * pool_sigmas[1] ** 2),
                               feed_flows[3] ** 2 * self.std ** 2 + feed_flows[4] ** 2 * self.std ** 2 -
                               (casadi.sum1(pool_flows[8:12]) ** 2 * pool_sigmas[2] ** 2),
                               feed_flows[5] ** 2 * self.std ** 2 - (
                                       casadi.sum1(pool_flows[12:16]) ** 2 * pool_sigmas[3] ** 2)
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Product mu balance
            g = casadi.vertcat(g,
                               casadi.sum1(pool_flows[[i * 4 for i in range(4)]] * pool_mus[[i for i in range(4)]]) -
                               product_flows[0] * product_mus[0],
                               casadi.sum1(
                                   pool_flows[[i * 4 + 1 for i in range(4)]] * pool_mus[[i for i in range(4)]]) -
                               product_flows[1] * product_mus[1],
                               casadi.sum1(
                                   pool_flows[[i * 4 + 2 for i in range(4)]] * pool_mus[[i for i in range(4)]]) -
                               product_flows[2] * product_mus[2],
                               casadi.sum1(
                                   pool_flows[[i * 4 + 3 for i in range(4)]] * pool_mus[[i for i in range(4)]]) -
                               product_flows[3] * product_mus[3]
                               )
            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Product sigma balance
            g = casadi.vertcat(g,
                               casadi.sum1(
                                   pool_flows[[i * 4 for i in range(4)]] ** 2 * pool_sigmas[
                                       [i for i in range(4)]] ** 2) -
                               product_flows[0] ** 2 * product_sigmas[0] ** 2,
                               casadi.sum1(
                                   pool_flows[[i * 4 + 1 for i in range(4)]] ** 2 * pool_sigmas[
                                       [i for i in range(4)]] ** 2)
                               - product_flows[1] ** 2 * product_sigmas[1] ** 2,
                               casadi.sum1(
                                   pool_flows[[i * 4 + 2 for i in range(4)]] ** 2 * pool_sigmas[
                                       [i for i in range(4)]] ** 2)
                               - product_flows[2] ** 2 * product_sigmas[2] ** 2,
                               casadi.sum1(
                                   pool_flows[[i * 4 + 3 for i in range(4)]] ** 2 * pool_sigmas[
                                       [i for i in range(4)]] ** 2)
                               - product_flows[3] ** 2 * product_sigmas[3] ** 2
                               )

            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Deriving the quality satisfaction probability from the cumulative distribution function
            for i in range(4):
                g = casadi.vertcat(g,
                                   0.5 * (1 + casadi.erf(
                                       (product_quality[i] - product_mus[i]) / (product_sigmas[i] * np.sqrt(2)))) -
                                   product_p[i]
                                   )
                lbg.append(0)
                ubg.append(np.inf)
                # ubg.append(0) ## Technically this should be a an equality constraint. In practice, the objective is monotonic in the probabilities so writing it as an inequality is valid.
        elif self.problem_name == 'Segarwak':
            pass

        # Now the local optimisation constrained to the particular sub-region of the feasible region can be completed
        problem = {'x': all_variables, 'f': objective, 'g': g}
        solver = casadi.nlpsol('solver', 'ipopt', problem, {'ipopt.tol': self.local_solver_tol})
        solution = solver(lbx=lb, ubx=ub, lbg=lbg, ubg=ubg)

        if solver.stats()['success']:
            print('This is an upper bound')
            print(solution['f'])
            return [solver.stats()['success'], float(solution['f']), solution['x']]
        else:
            print('This is an upper bound')
            print(np.inf)
            return [False, np.inf]

    def Get_erf_LB(self, product_quality=2.5, mu_bounds=[1.0, 3.0], sigma_bounds=[0.02, 0.2]):
        """
        Method to construct a convex relaxation of the error function as a wider goal of relaxing the cumulative
        distribution function of a normally distributed random variable. The mean and variance of the distribution
        are essentially the variables here and the measure of the interval they are defined over determines the
        tightness of the relaxation.
        """
        X, Y = np.meshgrid(np.linspace(mu_bounds[0], mu_bounds[1]), np.linspace(sigma_bounds[0], sigma_bounds[1]))

        Z = -0.5 * (1 + scp_erf((product_quality - mu_bounds[0]) / (Y * np.sqrt(2))))

        scalar_underestimate = np.min(Z)

        if mu_bounds[0] >= product_quality:

            f_lb = np.min(-0.5 * (1 + scp_erf((product_quality - mu_bounds[0]) / (Y * np.sqrt(2)))))
            f_ub = np.min(-0.5 * (1 + scp_erf((product_quality - mu_bounds[1]) / (Y * np.sqrt(2)))))

            slope = (f_lb - f_ub) / (mu_bounds[0] - mu_bounds[1])
        else:
            f_lb = np.min(-0.5 * (1 + scp_erf((product_quality - mu_bounds[0]) / (Y * np.sqrt(2)))))
            slopes = (0.5 * np.sqrt(2 / np.pi) * np.exp(-((product_quality - mu_bounds[0]) ** 2) / (2 * Y ** 2))) / Y
            slope = np.min(slopes)

        return scalar_underestimate, [f_lb, slope]

    def ss_evaluator(self):
        """
        Since the approach already works directly with our validation model, this does not need to be reconstructed and
        the "stochastic solution" is simply given as the solution we obtained in our optimisation.
        :return:
        """
        if self.problem_status == 'unsolved':
            print('Problem has not been sovled using an optimisation under uncertainty method yet.')
            print('Optimise the problem using one of the available methods first.')

            exit()
        else:
            self.stochastic_solution = self.continuous_solution

    def save_results(self):
        """
        A method to save the results to a csv result file. First the method checks if a results file for the appropriate
         problem/solution approach exists, and if it does it pull it and only updates the field for the relevant
         combination of true uncertainty variance and scenario generation choices. It then saves the file. If no
         appropriate file exists initially, it creates one from scratch.
        """
        try:
            dataframe = pd.read_csv(self.problem_name + '_stochastic_programming_results.csv', index_col=['std'])
        except:
            index = pd.Index(data=[], name='std')
            dataframe = pd.DataFrame(index=index, columns=['f', 'best_lb', 'CPU', 'CPU_to_solution'])

        dataframe.loc[float(self.std), :] = [self.continuous_solution, self.continuous_bound, self.runTime,
                                             self.runTimetoSol]

        dataframe = dataframe.sort_index(ascending=False)

        dataframe.to_csv(self.problem_name + '_stochastic_programming_results.csv')

