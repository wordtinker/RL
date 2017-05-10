using System;
using Accord.Statistics.Distributions.Univariate;
using System.Linq;

namespace MultiArmBandit
{
    abstract class ABandit
    {
        public abstract int Size { get; }
        public abstract double this[int index] { get; }
        public abstract double Reward(int index);
    }

    // Stationary bandit
    class KArmBandit : ABandit
    {
        private double[] arms;

        // Number of actions
        public override int Size
        {
            get { return arms.Length;  }
        }

        // the action values q∗(a)
        public override double this[int index]
        {
            get { return arms[index]; }
        }

        // Reward function
        public override double Reward(int index)
        {
            // Create normal distribution with mean=0 and variance 1
            NormalDistribution norm = new NormalDistribution(mean: this[index], stdDev: 1);
            return norm.Generate();
        }

        public KArmBandit(int nOfArms)
        {
            // Create normal distribution with mean=0 and variance 1
            NormalDistribution norm = new NormalDistribution(mean: 0, stdDev: 1);
            // Seed k armed bandit
            arms = norm.Generate(nOfArms);
        }
    }

    // NonStationary bandit
    class MovingKArmBandit : ABandit
    {
        private double[] arms;
        private double[] weights;

        // Number of actions
        public override int Size
        {
            get { return arms.Length; }
        }

        // the action values q∗(a)
        public override double this[int index]
        {
            get { return arms[index]; }
        }

        // Reward function
        public override double Reward(int index)
        {
            // On every call move every action value a bit
            for (int i = 0; i < arms.Length; i++)
            {
                arms[i] += weights[i];
            }

            // Create normal distribution with mean=0 and variance 1
            NormalDistribution norm = new NormalDistribution(mean: this[index], stdDev: 1);
            return norm.Generate();
        }

        public MovingKArmBandit(int nOfArms)
        {
            // Create normal distribution with mean=0 and variance 1
            NormalDistribution norm = new NormalDistribution(mean: 0, stdDev: 1);
            // Seed k armed bandit
            arms = norm.Generate(nOfArms);
            // for each arm generate either -1 or +1
            // giving it direction and weight
            Random rnd = new Random();
            weights = new double[nOfArms];
            for (int i = 0; i < nOfArms; i++)
            {
                weights[i] = (rnd.Next(2) * 2 - 1) * 0.001 * arms[i];
            }
        }
    }

    class EpsilonGreedy
    {
        private double epsilon;
        // estimate of each action
        private double[] Q;
        // Number of tries for each action
        private int[] N;

        public double Estimate(int index)
        {
            return Q[index];
        }

        public int Seen(int index)
        {
            return N[index];
        }

        public void Run(ABandit bandit, int times)
        {
            // initialize
            Q = new double[bandit.Size];
            N = new int[bandit.Size];
            Random rnd = new Random();
            // 
            for (int i = 0; i < times; i++)
            {
                int action;
                if (rnd.NextDouble() > epsilon)
                {
                    // exploitation
                    // chose maximum yielding estimate
                    // TODO optimize
                    double maxValue = Q.Max();
                    action = Q.ToList().IndexOf(maxValue);
                }
                else
                {
                    // exploration
                    // chose random action
                    action = rnd.Next(bandit.Size);
                }
                double reward = bandit.Reward(action);
                N[action] += 1;
                Q[action] += (reward - Q[action]) / (N[action]);
            }
        }

        public EpsilonGreedy(double epsilon)
        {
            this.epsilon = epsilon;
        }
    }

    class WeightedEpsilonGreedy
    {
        // weighting parameter
        private double alpha;
        private double epsilon;
        // estimate of each action
        private double[] Q;
        // Number of tries for each action
        private int[] N;

        public double Estimate(int index)
        {
            return Q[index];
        }

        public int Seen(int index)
        {
            return N[index];
        }

        public void Run(ABandit bandit, int times)
        {
            // Method is generallly the same as in EpsilonGreedy
            // but will not converge to the action value.

            // initialize
            Q = new double[bandit.Size];
            N = new int[bandit.Size];
            Random rnd = new Random();
            // 
            for (int i = 0; i < times; i++)
            {
                int action;
                if (rnd.NextDouble() > epsilon)
                {
                    // exploitation
                    // chose maximum yielding estimate
                    // TODO optimize
                    double maxValue = Q.Max();
                    action = Q.ToList().IndexOf(maxValue);
                }
                else
                {
                    // exploration
                    // chose random action
                    action = rnd.Next(bandit.Size);
                }
                double reward = bandit.Reward(action);
                N[action] += 1;
                Q[action] += alpha * (reward - Q[action]);
            }
        }

        public WeightedEpsilonGreedy(double epsilon, double alpha)
        {
            this.epsilon = epsilon;
            this.alpha = alpha;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Stationary problem
            // Create one k-arm bandit
            var bandit = new KArmBandit(10);
            // Create ε-greedy method
            var method = new EpsilonGreedy(0.1);
            method.Run(bandit, 1000);
            var alphaMethod = new WeightedEpsilonGreedy(0.1, 0.1);
            alphaMethod.Run(bandit, 1000);

            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Action value: {bandit[i]:N2}, Estimate: {method.Estimate(i):N2}, Seen: {method.Seen(i)}");
            }
            Console.WriteLine();
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Action value: {bandit[i]:N2}, Estimate: {alphaMethod.Estimate(i):N2}, Seen: {alphaMethod.Seen(i)}");
            }

            // Nonstationary problem
            var movingBandit = new MovingKArmBandit(10);
            Console.WriteLine();
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Starting action value: {movingBandit[i]:N2}");
            }
            method.Run(movingBandit, 1000);
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Action value: {movingBandit[i]:N2}, Estimate: {method.Estimate(i):N2}, Seen: {method.Seen(i)}");
            }
            Console.WriteLine();
            movingBandit = new MovingKArmBandit(10);
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Starting action value: {movingBandit[i]:N2}");
            }
            alphaMethod.Run(movingBandit, 1000);
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Action value: {movingBandit[i]:N2}, Estimate: {alphaMethod.Estimate(i):N2}, Seen: {alphaMethod.Seen(i)}");
            }

            // TODO
            // stochastic gradient ascent method (see 2.7 Reinforcement Learning: An Introduction. Sutton. Barto )
        }
    }
}
