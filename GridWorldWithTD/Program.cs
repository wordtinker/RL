using System;
using System.Collections.Generic;
using System.Linq;

namespace GridWorldWithTD
{
    class PolicyState
    {
        private Random rnd;
        // state value
        public double Value { get; set; }
        // state return
        public double G { get; set; }
        // number of times state has been visited(first time)
        public int N { get; set; }
        // list of actions
        public Dictionary<Action, StateActionPolicy> Actions { get; }
        public void Rebalance(double epsilon = 0.2)
        {
            double max = Actions.Values.Max(p => p.Value);
            double size = Actions.Where(kvp => kvp.Value.Value == max).Count();
            if (epsilon == 0)
            {
                // policy is greedy
                foreach (var item in Actions)
                {
                    item.Value.P = item.Value.Value == max ? 1 / size : 0;
                }
            }
            else
            {
                // soft policy
                double x = (1 - epsilon) / size + epsilon / (double)Actions.Count;
                foreach (var item in Actions)
                {
                    item.Value.P = item.Value.Value == max ? x : epsilon / (double)Actions.Count;
                }
            }
        }
        public Action NextAction
        {
            get
            {
                // choose action randomly
                // any policy is supported(greedy, soft-epsilon)
                double ra = rnd.NextDouble();
                double accumulator = 0;
                foreach (var kvp in Actions.Where(sap => sap.Value.InPolicy))
                {
                    accumulator += kvp.Value.P;
                    if (ra < accumulator)
                    {
                        return kvp.Key;
                    }
                }
                // normally it would have returned already, but if
                // due to uneven division of probabilities
                // ra was ever less than accumulated probability
                // return last key
                return Actions.Keys.Last();
            }
        }
        public PolicyState(State s)
        {
            rnd = new Random();
            Actions = new Dictionary<Action, StateActionPolicy>();
            double size = s.Actions.Count();
            foreach (Action a in s.Actions)
            {
                Actions[a] = new StateActionPolicy();
                Actions[a].P = 1 / size;
            }
        }
    }

    class StateActionPolicy
    {
        // state action value
        public double Value { get; set; }
        // cumulative sum Cn of the weights given to the first n returns for off policy learning
        public double C { get; set; }
        // state action return
        public double G { get; set; }
        // number of times state action has been visited(first time or every time)
        public int N { get; set; }
        // marks action as used by Policy
        public bool InPolicy { get { return P > 0; } }
        // probability of activation in policy
        public double P { get; set; }

        public StateActionPolicy()
        {
            Value = 0;
        }
    }

    static class Walker
    {
        private static Random rnd = new Random();
        private static IEnumerable<Tuple<State, Action, double>> GenerateWalk(AEnvironment env, Policy p, State s, Action a)
        {
            // Reward
            double r;
            while (true)
            {
                // ask for new state and reward
                var nsr = env.GetReward(s, a);
                r = nsr.Item2;

                yield return Tuple.Create(s, a, r);

                s = nsr.Item1;
                if (s.Type == StateType.Terminal)
                {
                    break;
                }
                // choose action by the policy
                a = p.P[s].NextAction;
            }
        }

        /// <summary>
        /// Generates an episode starting from S0, A0, following π
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<Tuple<State, Action, double>> GenerateEpisode(AEnvironment env, Policy p, bool exploringStart = true)
        {
            int startingState = rnd.Next(env.States.Count());
            State s = env.States.ElementAt(startingState);
            Action a;
            if (exploringStart)
            {
                // Exploring start
                // any s,a pair is viable as starting pair
                // regardless of policy
                int startingAction = rnd.Next(s.Actions.Count());
                a = s.Actions.ElementAt(startingAction);
            }
            else
            {
                // choose action by the policy
                a = p.P[s].NextAction;
            }
            // Generate a walk from that position
            foreach (var item in GenerateWalk(env, p, s, a))
            {
                yield return item;
            }
        }
    }

    class Bank
    {
        private int limit;
        private Queue<Tuple<State, Action, double>> queue;

        public void Enqueue(Tuple<State, Action, double> sar)
        {
            if (queue.Count >= limit) throw new Exception();
            queue.Enqueue(sar);
        }
        public Tuple<State, Action, double> CalculateG(double gamma)
        {
            double g = queue.Select((n, i) => Math.Pow(gamma, i) * n.Item3).Sum();
            var sar = queue.Dequeue();
            return Tuple.Create(sar.Item1, sar.Item2, g);
        }

        public Bank(int limit)
        {
            this.limit = limit;
            this.queue = new Queue<Tuple<State, Action, double>>();
        }
    }

    enum PolicyMethod
    {
        SARSA,
        QLearning,
        ExpectedSARSA
    }

    class Policy
    {
        public Dictionary<State, PolicyState> P { get; }
        private AEnvironment env;
        private double gamma;
        private Random rnd;

        public double V(State s)
        {
            return P[s].Value;
        }

        public double V(State s, Action a)
        {
            return P[s].Actions[a].Value;
        }

        public Dictionary<Action, StateActionPolicy> A(State s)
        {
            return P[s].Actions;
        }

        public int Learn(PolicyMethod method, int limit = 1000, double alpha = 0.5)
        {
            for (int i = 0; i < limit; i++)
            {
                // Generate an episode
                // episode MUST be IEnumerable so policy can be evaluated
                // at runtime
                var episode = Walker.GenerateEpisode(env, this, exploringStart: false);
                var enumer = episode.GetEnumerator();
                // move to first element
                enumer.MoveNext();
                // chose A from S using policy
                State s = enumer.Current.Item1;
                Action a = enumer.Current.Item2;
                double r = enumer.Current.Item3;
                // for every action next in the episode
                foreach (var sar in episode)
                {
                    // observe next state and next action using policy
                    State nextState = sar.Item1;
                    Action nextAction = sar.Item2;
                    // Update Q for s-a
                    StateActionPolicy sap = P[s].Actions[a];
                    double currentValue = sap.Value;
                    // chose method of learning
                    double nextSAValue;
                    switch (method)
                    {
                        case PolicyMethod.SARSA:
                            // get the value on next action
                            nextSAValue = P[nextState].Actions[nextAction].Value;
                            break;
                        case PolicyMethod.QLearning:
                            // get the maximum value of next actions 
                            nextSAValue = P[nextState].Actions.Values.Max(p => p.Value);
                            break;
                        case PolicyMethod.ExpectedSARSA:
                            // get the expected value of the next state
                            nextSAValue = P[nextState].Actions.Values.Select(p => p.P * p.Value).Sum();
                            break;
                        default:
                            throw new NotImplementedException();
                    }
                    sap.Value = currentValue + alpha * (r + gamma * nextSAValue - currentValue);
                    sap.N++;
                    // rebalance probabilities of current policy
                    P[s].Rebalance(epsilon: 0.2);
                    s = nextState;
                    a = nextAction;
                    r = sar.Item3;
                }
            }
            return limit;
        }

        public int NStepSARSA(int limit = 1000, double alpha = 0.5, int N = 5)
        {
            int T;
            for (int j = 0; j < limit; j++)
            {
                // Generate an episode
                // episode MUST be IEnumerable so policy can be evaluated
                // at runtime
                var episode = Walker.GenerateEpisode(env, this, exploringStart: false);
                var enumer = episode.GetEnumerator();
                int tau;
                Bank bank = new Bank(N);
                // move to first element
                enumer.MoveNext();
                // chose A from S using policy
                State s = enumer.Current.Item1;
                Action a = enumer.Current.Item2;
                T = int.MaxValue;
                // for each next state
                for (int t = 0;; t++)
                {
                    if (t < T)
                    {
                        // Store Ai and Ri+1
                        bank.Enqueue(enumer.Current);
                        // check if we reached the end of episode
                        if (!enumer.MoveNext())
                        {
                            T = t + 1;
                        }
                        else
                        {
                            // chose A from S using policy
                            s = enumer.Current.Item1;
                            a = enumer.Current.Item2;
                        }
                    }
                    // the time whose estimate is being updated
                    tau = t - N + 1;
                    if (tau >= 0)
                    {
                        // calculate return
                        Tuple<State, Action, double> sag =  bank.CalculateG(gamma);
                        StateActionPolicy sap = P[sag.Item1].Actions[sag.Item2];
                        double G = sag.Item3;
                        // calculate estimate of future value tail
                        if (tau + N < T) G += Math.Pow(gamma, N) * P[s].Actions[a].Value;
                        sap.Value += alpha * (G - sap.Value);
                        // update policy
                        P[sag.Item1].Rebalance(epsilon: 0.2);
                    }
                    // end loop condition
                    if (tau == T - 1) break;
                }
            }
            return limit;
        }

        public Policy(AEnvironment env, double gamma = 1)
        {
            this.rnd = new Random();
            this.env = env;
            this.gamma = gamma;
            // create equiprobable random policy
            P = new Dictionary<State, PolicyState>();
            foreach (State s in env.StatesPlus)
            {
                P[s] = new PolicyState(s);
            }
        }
    }

    class Program
    {
        static void NStep(AEnvironment env)
        {
            Console.WriteLine("Initial gridworld");
            env.Print();
            Console.WriteLine();

            Policy p = new Policy(env, gamma: 0.5);
            Console.WriteLine("Starting policy");
            env.PrintPolicy(p);
            Console.WriteLine();

            p.NStepSARSA(limit: 50000, alpha: 0.5, N: 5);
            env.PrintPolicy(p);
            env.PrintsSAV(p);
            // show as greedy policy
            foreach (var item in p.P)
            {
                if (item.Key.Type != StateType.Terminal)
                {
                    item.Value.Rebalance(0);
                }
            }
            env.PrintPolicy(p);
        }
        static void OneStep(AEnvironment env, PolicyMethod method)
        {
            Console.WriteLine("Initial gridworld");
            env.Print();
            Console.WriteLine();

            Policy p = new Policy(env, gamma: 0.5);
            Console.WriteLine("Starting policy");
            env.PrintPolicy(p);
            Console.WriteLine();

            p.Learn(method, limit: 100000, alpha: 0.5);
            env.PrintPolicy(p);
            env.PrintsSAV(p);
            // show as greedy policy
            foreach (var item in p.P)
            {
                if (item.Key.Type != StateType.Terminal)
                {
                    item.Value.Rebalance(0);
                }
            }
            env.PrintPolicy(p);
        }
        static void Main(string[] args)
        {
            Console.WriteLine("1 - SARSA(On-policy) on Grid World");
            Console.WriteLine("2 - SARSA(On-policy) on Windy World");
            Console.WriteLine("3 - Q-Learning(Off-policy) on Grid World");
            Console.WriteLine("4 - Expected SARSA(Off-policy) on Grid World");
            Console.WriteLine("5 - n-SARSA on Windy World");
            string decision = Console.ReadLine();
            AEnvironment env;
            PolicyMethod method;
            switch (decision)
            {
                case "1":
                    env = new GridWorld();
                    method = PolicyMethod.SARSA;
                    OneStep(env, method);
                    break;
                case "2":
                    env = new WindyGrid();
                    method = PolicyMethod.SARSA;
                    OneStep(env, method);
                    break;
                case "3":
                    env = new GridWorld();
                    method = PolicyMethod.QLearning;
                    OneStep(env, method);
                    break;
                case "4":
                    env = new GridWorld();
                    method = PolicyMethod.ExpectedSARSA;
                    OneStep(env, method);
                    break;
                case "5":
                    env = new WindyGrid();
                    NStep(env);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }
    }
}
