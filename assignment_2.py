"""Assignment 2 Financial Mathematics"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class Assignment2:
    def __init__(self):
        # risk free asset return
        self._rf = 0.005

        # read data
        self._history = pd.read_csv('history.csv')
        self._future = pd.read_csv('future.csv')

        # change index, calculate returns and drop the first row 
        # (since we have n - 1 returns for n days)
        self._returns = \
            self._history.set_index('date').pct_change().drop('2/25/2013')

        # set the gains
        self._gains = np.log(self._returns + 1)

    def task_1(self):
        print(self._returns.mean() * 252)
        print(self._returns.var() * 252)
        print(self._returns.cov() * 252)

    def task_2(self):
        # generate the mean return per stock (for a year)
        means = np.asarray(self._returns.mean() * 252)

        # generate the covariance matrix and its inverse;
        # also generate the row of ones
        cov_matrix = (self._returns.cov() * 252).as_matrix()
        inverse_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.asarray([1] * len(means))

        # weights, mean and var of the minimum variances portofolio
        weights_min = np.dot(
            ones, inverse_cov_matrix) / np.dot(
                np.dot(ones, inverse_cov_matrix), ones)

        print("Minimum variance portofolio weights:")
        for weight in weights_min:
            print(weight, end=' ')

        print('\n')

        min_portofolio = self.get_mean_var(weights_min, means, cov_matrix, 'min')

        print('Mean:', min_portofolio[0])
        print('Var:', min_portofolio[1])

        # weights of the tangency portofolio
        weights_tan = np.dot(
            means - self._rf, inverse_cov_matrix) / np.dot(
                np.dot(means - self._rf, inverse_cov_matrix), ones)
        print('Tangency portofolio:')

        for weight in weights_tan:
            print(weight, end=' ')

        print('\n')

        tan_portofolio = self.get_mean_var(weights_tan, means, cov_matrix, 'tan')

        print('Mean:', tan_portofolio[0])
        print('Var:', tan_portofolio[1])

        # visualize the portofolios, the effcient frontier and the CML
        num_assets = len(means)
        num_portofolios = 30000
        port_returns = []
        port_stds = []

        for _ in range(num_portofolios):
            weights = np.random.uniform(-1.0, 1.0, num_assets)
            weights /= np.sum(weights)
            port_returns.append(np.dot(weights, means))
            port_stds.append(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))

        portofolio = {'Returns': port_returns,
                      'Stds': port_stds}

        port = pd.DataFrame(portofolio)
        mask = ((port['Returns'] <= 1.0) & (port['Returns'] >= -0.3)) & (port['Stds'] <= 0.6)
        port = port[mask]

        plt.style.use('seaborn')
        # feasible region
        ax = port.plot.scatter(x='Stds', y='Returns', figsize=(10, 8), grid=True)
        # tangency portofolio
        plt.scatter(x=np.sqrt(tan_portofolio[1]), y=tan_portofolio[0], c='red', marker='D', s=150)
        # minimum variance portofolio
        plt.scatter(x=np.sqrt(min_portofolio[1]), y=min_portofolio[0], c='blue', marker='D', s=150)
        # risk free asset
        plt.scatter(x=0, y=self._rf, c='green', marker='D', s=200)
        # capital market line
        cpm_means = []
        cpm_stds = []

        for _ in range(10000):
            std = np.random.uniform(0, 0.6)
            cpm_stds.append(std)
            cpm_means.append(self._rf + std * (
                tan_portofolio[0] - self._rf) / np.sqrt(tan_portofolio[1]))

        cpm = {'Returns': cpm_means,
               'Stds': cpm_stds}

        cpm_df = pd.DataFrame(cpm)
        cpm_df.plot.scatter(ax=ax, x='Stds', y='Returns', c='black', s=0.25)

        plt.xlabel('Std. Deviation')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier and CPM')
        plt.show()

    def task_3(self):
        """Task 3: avearge of the min and tan portofolios"""
        # get the means
        means = np.asarray(self._returns.mean() * 252)
        # get the covariance matrix and the inverse
        cov_matrix = (self._returns.cov() * 252).as_matrix()
        inverse_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.asarray([1] * len(means))

        # weights of the minimum variances portofolio
        weights_min = np.dot(
            ones, inverse_cov_matrix) / np.dot(
                np.dot(ones, inverse_cov_matrix), ones)
        # weights of the tangency portofolio
        weights_tan = np.dot(
            means - self._rf, inverse_cov_matrix) / np.dot(
                np.dot(means - self._rf, inverse_cov_matrix), ones)
        # weights of the averaged portofolio
        weights_port = (weights_min + weights_tan) / 2

        # calculation of the returns and gains
        self._history = self._history.set_index('date')
        self._history['Returns'] = np.dot(self._history, weights_port)
        self._history = self._history.pct_change().drop('2/25/2013')
        self._history['Gains'] = np.log(self._history['Returns'] + 1)

        # histograms of the returns and gains
        self._history.hist(column='Returns', bins=100)
        plt.title('Returns Histogram')
        self._history.hist(column='Gains', bins=100)
        plt.title('Gains Histogram')
        plt.show()

    def task_4(self):
        """Task 3: avearge of the min and tan portofolios"""
        # get the means
        means = np.asarray(self._returns.mean() * 252)
        # get the covariance matrix and the inverse
        cov_matrix = (self._returns.cov() * 252).as_matrix()
        inverse_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.asarray([1] * len(means))

        # weights of the minimum variances portofolio
        weights_min = np.dot(
            ones, inverse_cov_matrix) / np.dot(
                np.dot(ones, inverse_cov_matrix), ones)
        # weights of the tangency portofolio
        weights_tan = np.dot(
            means - self._rf, inverse_cov_matrix) / np.dot(
                np.dot(means - self._rf, inverse_cov_matrix), ones)
        # weights of the averaged portofolio
        weights_port = (weights_min + weights_tan) / 2

        # calculation of the returns and gains
        self._history = self._history.set_index('date')
        self._history['Returns'] = np.dot(self._history, weights_port)
        self._history = self._history.pct_change().drop('2/25/2013')
        self._history['Gains'] = np.log(self._history['Returns'] + 1)

        # compute the mean return per year, and the variance per year for history.csv
        portfolio_mean_return_history = 252 * self._history['Returns'].mean()
        portfolio_variance_history = 252 * self._history['Returns'].var()

        # compute the sharpe ratio for history.csv
        sharpe_ratio_history = (portfolio_mean_return_history - self._rf) / sqrt(portfolio_variance_history)

        print("Sharpe ratio for the portfolio (history.csv) = ", sharpe_ratio_history)

        # calculate returns for future.csv
        self._future = self._future.set_index('date')
        self._future['Returns'] = np.dot(self._future, weights_port)
        self._future = self._future.pct_change().drop('1/2/2018')

        # compute the mean return per year, and the variance per year for future.csv
        portfolio_mean_return_future = 252 * self._future['Returns'].mean()
        portfolio_variance_future = 252 * self._history['Returns'].var()
        
        # compute the sharpe ratio for future.csv
        sharpe_ratio_future = (portfolio_mean_return_future - self._rf) / sqrt(portfolio_variance_future)

        print("Sharpe ratio for the portfolio (future.csv) = ", sharpe_ratio_future)

        # compute the value at risk
        
    def get_mean_var(self, weights, means, cov_matrix, portofolio):
        """get the mean and variance based on the portofolio"""
        inverse_cov_matrix = np.linalg.inv(cov_matrix)
        mean = np.dot(weights, means)
        ones = np.asarray([1] * len(means))
        if portofolio is 'min':
            var = 1 / sum(np.dot(ones, inverse_cov_matrix))
        elif portofolio is 'tan' or portofolio is 'other':
            var = np.dot(weights, np.dot(cov_matrix, weights))
        return (mean, var)

    def _mean(self, stock):
        summation = 0
        counter = 0
        for val in self._returns[stock]:
            summation += val
            counter += 1
        return summation / counter

    def _var(self, stock):
        summation = 0
        counter = 0
        mean = self._mean(stock)

        for val in self._returns[stock]:
            summation += (val - mean)**2
            counter += 1

        return summation / (counter - 1)

    def _cov(self, stock1, stock2):
        summation = 0
        counter = 0

        mean1 = self._mean(stock1)
        mean2 = self._mean(stock2)

        for val1, val2 in zip(self._returns[stock1], self._returns[stock2]):
            summation += (val1 - mean1) * (val2 - mean2)
            counter += 1

        return summation / (counter - 1)

#Assignment2().task_1()
#Assignment2().task_2()
#Assignment2().task_3()
Assignment2().task_4()