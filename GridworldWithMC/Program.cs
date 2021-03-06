﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace GridworldWithDP
{
    enum Action
    {
        Up, Down, Left, Right
    }

    enum StateType
    {
        Terminal, NonTerminal
    }

    class State
    {
        internal int i;
        internal int j;
        public StateType Type { get; internal set; }

        public IEnumerable<Action> Actions
        {
            get
            {
                if (Type == StateType.NonTerminal)
                {
                    foreach (var item in Enum.GetValues(typeof(Action)).Cast<Action>())
                    {
                        yield return item;
                    }
                }
                else
                {
                    yield break;
                }
            }
        }
        public State(int i, int j, StateType type = StateType.NonTerminal)
        {
            this.i = i;
            this.j = j;
            this.Type = type;
        }
    }

    class Environment
    {
        State[][] states;

        public IEnumerable<State> StatesPlus
        {
            get
            {
                foreach (var item in states)
                {
                    foreach (State s in item)
                    {
                        yield return s;
                    }
                }
            }
        }

        public IEnumerable<State> States
        {
            get
            {
                foreach (var item in states)
                {
                    foreach (State s in item)
                    {
                        if (s.Type != StateType.Terminal) yield return s;
                    }
                }
            }
        }

        /// <summary>
        /// Hidden information.
        /// </summary>
        /// <param name="state"></param>
        /// <param name="action"></param>
        /// <returns></returns>
        private State NextState(State state, Action action)
        {
            try
            {
                switch (action)
                {
                    // probability of every action is 100%
                    case Action.Up:
                        return states[state.i - 1][state.j];
                    case Action.Down:
                        return states[state.i + 1][state.j];
                    case Action.Left:
                        return states[state.i][state.j - 1];
                    case Action.Right:
                        return states[state.i][state.j + 1];
                    default:
                        return state;
                }
            }
            catch (Exception)
            {
                return state;
            }
        }

        /// <summary>
        /// Do not have to know next state, not like in DP solution.
        /// </summary>
        /// <param name="state"></param>
        /// <param name="action"></param>
        /// <returns>New stat and reward </returns>
        public Tuple<State, double> GetReward(State state, Action action)
        {
            State newState = NextState(state, action);
            if (newState.Type == StateType.Terminal)
            {
                return Tuple.Create(newState, 10.00);
            }
            else if (state == newState)
            {
                return Tuple.Create(newState, -2.00);
            }
            return Tuple.Create(newState, -1.00);
        }

        public void Print()
        {
            foreach (var item in states)
            {
                foreach (State s in item)
                {
                    if (s.Type == StateType.NonTerminal)
                    {
                        Console.Write("- ");
                    }
                    else
                    {
                        Console.Write("* ");
                    }
                }
                Console.WriteLine();
            }
        }

        public void PrintVs(Policy policy)
        {
            foreach (var item in states)
            {
                foreach (State s in item)
                {
                    Console.Write($"{policy.V(s):N2} ");
                }
                Console.WriteLine();
            }
        }

        public void PrintsSAV(Policy policy)
        {
            foreach (var s in States)
            {
                foreach (var kvp in policy.A(s))
                {
                    Console.WriteLine($"{s.i} {s.j} {kvp.Key} {kvp.Value.Value:N2} {kvp.Value.N}");
                }
                Console.WriteLine();
            }
        }

        public void PrintPolicy(Policy policy)
        {
            foreach (var item in states)
            {
                string line1 = "";
                string line2 = "";
                string line3 = "";
                string line4 = "";
                string line5 = "";
                foreach (State s in item)
                {
                    if (s.Type == StateType.Terminal)
                    {
                        line1 += "xxxxxxxxxxx ";
                        line2 += "xxxxxxxxxxx ";
                        line3 += "xxxxxxxxxxx ";
                        line4 += "xxxxxxxxxxx ";
                        line5 += "xxxxxxxxxxx ";
                    }
                    else
                    {
                        string empty = "            ";
                        var actions = policy.A(s);
                        var partial = actions.Where(kvp => kvp.Value.InPolicy).Select(kvp => kvp.Key);
                        line1 += partial.Contains(Action.Up) ? "     ^      " : empty;
                        line2 += partial.Contains(Action.Up) ? $"    {actions[Action.Up].P:N2}    " : empty;
                        line3 += partial.Contains(Action.Left) ? $"<{actions[Action.Left].P:N2} " : "      ";
                        line3 += partial.Contains(Action.Right) ? $" {actions[Action.Right].P:N2}>" : "      ";
                        line4 += partial.Contains(Action.Down) ? $"    {actions[Action.Down].P:N2}    " : empty;
                        line5 += partial.Contains(Action.Down) ? "     v      " : empty;
                    }
                }
                Console.WriteLine(line1);
                Console.WriteLine(line2);
                Console.WriteLine(line3);
                Console.WriteLine(line4);
                Console.WriteLine(line5);
                Console.WriteLine();
            }
        }

        public Environment()
        {
            states = new State[4][];
            for (int i = 0; i < 4; i++)
            {
                states[i] = new State[4];
                for (int j = 0; j < 4; j++)
                {
                    states[i][j] = new State(i, j);
                }
            }
            states[0][0].Type = StateType.Terminal;
            states[3][3].Type = StateType.Terminal;
        }
    }

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
        private static IEnumerable<Tuple<State, Action, double>> GenerateWalk(Environment env, Policy p, State s, Action a)
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
        public static List<Tuple<State, Action, double>> GenerateEpisode(Environment env, Policy p, bool exploringStart = true)
        {
            List<Tuple<State, Action, double>> episode = new List<Tuple<State, Action, double>>();
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
                episode.Add(item);
            }
            return episode;
        }
    }

    class Policy
    {
        public Dictionary<State, PolicyState> P { get; }
        private Environment env;
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

        /// <summary>
        /// First-visit MC
        /// TODO Can't generate policy from that w/o proper model.
        /// </summary>
        /// <param name="limit"></param>
        /// <returns></returns>
        public int EvaluateMC(int limit = 1000)
        {
            // initialize V(s) = 0 for all states
            foreach (PolicyState ps in P.Values)
            {
                ps.Value = 0;
            }

            for (int i = 0; i < limit; i++)
            {
                // Generate an episode
                var episode = Walker.GenerateEpisode(env, this);
                // initiate new return G
                double g = 0;
                foreach (var sar in episode.Reverse<Tuple<State, Action, double>>())
                {
                    // calculate return for current state
                    // discounted by gamma
                    g = sar.Item3 + gamma * g;
                    // we are running in reverse order so
                    // return will be properly propagated
                    // to state that has been visited first
                    P[sar.Item1].G = g;
                }
                foreach (var sr in episode)
                {
                    // update value function
                    PolicyState p = P[sr.Item1];
                    p.Value += (p.G - p.Value) / (double)++p.N;
                    p.G = 0;
                }
            }
            return limit;
        }

        /// <summary>
        /// TODO works weird with s-a pair with equal
        /// probabilities, might consider adding variance
        /// as approval.
        /// </summary>
        /// <param name="limit"></param>
        /// <returns></returns>
        public int IterateMC(int limit = 1000)
        {
            for (int i = 0; i < limit; i++)
            {
                // Generate an episode
                var episode = Walker.GenerateEpisode(env, this);

                // prediction
                // initiate new return G
                double g = 0;
                foreach (var sar in episode.Reverse<Tuple<State, Action, double>>())
                {
                    // calculate return for current state action pair
                    // discounted by gamma
                    g = sar.Item3 + gamma * g;
                    // we are running in reverse order so
                    // return will be properly propagated
                    // to state action pair that has been visited first
                    P[sar.Item1].Actions[sar.Item2].G = g;
                }
                foreach (var sar in episode)
                {
                    // update value function
                    StateActionPolicy sap = P[sar.Item1].Actions[sar.Item2];
                    sap.Value += (sap.G - sap.Value) / (double)++sap.N;
                }
            }
            // control
            foreach (var item in P)
            {
                if (item.Key.Type != StateType.Terminal)
                {
                    // greedily mark only max yielding actions
                    item.Value.Rebalance(0);
                }
            }
            return limit;
        }

        public int OnPolicyFirstVisitMCControl(int limit = 1000, double epsilon = 0.1)
        {
            for (int i = 0; i < limit; i++)
            {
                // Generate an episode
                var episode = Walker.GenerateEpisode(env, this, exploringStart: false);

                // prediction
                // initiate new return G
                double g = 0;
                foreach (var sar in episode.Reverse<Tuple<State, Action, double>>())
                {
                    // calculate return for current state action pair
                    // discounted by gamma
                    g = sar.Item3 + gamma * g;
                    // we are running in reverse order so
                    // return will be properly propagated
                    // to state action pair that has been visited first
                    P[sar.Item1].Actions[sar.Item2].G = g;
                }
                foreach (var sar in episode)
                {
                    State s = sar.Item1;
                    // update value function
                    StateActionPolicy sap = P[s].Actions[sar.Item2];
                    sap.Value += (sap.G - sap.Value) / (double)++sap.N;
                    // control
                    // softly mark only max yielding actions
                    P[s].Rebalance(epsilon);
                }
            }
            return limit;
        }

        public int OffPolicyEveryVisitMCControl(Policy behaviour, int limit = 1000)
        {
            // Initialize for every state action pair
            var saps = from s in P.Values
                    from sap in s.Actions.Values
                    select sap;
            foreach (StateActionPolicy sap in saps)
            {
                sap.C = 0;
            }

            for (int i = 0; i < limit; i++)
            {
                // Generate an episode using behavior policy
                var episode = Walker.GenerateEpisode(env, behaviour, exploringStart: false);
                // initiate new return G
                double g = 0;
                // initiate importance sampling ratio
                double w = 1;
                foreach (var sar in episode.Reverse<Tuple<State, Action, double>>())
                {
                    State s = sar.Item1;
                    Action a = sar.Item2;
                    double r = sar.Item3;

                    // prediction
                    // calculate return for current state action pair
                    // discounted by gamma
                    g = r + gamma * g;
                    // we are running in reverse order so
                    // return will be properly propagated
                    // to state action pair that has been visited first
                    StateActionPolicy sap = P[s].Actions[a];
                    // update cumulative sum of ratios
                    sap.C += w;
                    // update value of state action pair
                    sap.Value += w / sap.C * (g - sap.Value);
                    // just for tracking
                    sap.N++;

                    // control
                    // greedily mark only max yielding actions
                    P[s].Rebalance(0);

                    // termination condition
                    // if taget policy already knows best s-a pair for that step
                    // in episode after update, there is no reason
                    // to follow behavior generated episode farther
                    var bestActions = from kvp in P[s].Actions
                                      where kvp.Value.InPolicy
                                      select kvp.Key;
                    if (!bestActions.Contains(a))
                    {
                        break;
                    }
                    // update ratio
                    // as the policy is soft we have to use probability of target policy
                    // w will be >= 1
                    w = w * P[s].Actions[a].P / behaviour.P[s].Actions[a].P;
                }
            }
            return limit;
        }

        public Policy(Environment env, double gamma = 1)
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
        static void EvaluatePolicyWithMC()
        {
            Environment env = new Environment();
            Console.WriteLine("Initial gridworld");
            env.Print();
            Console.WriteLine();

            Policy policy = new Policy(env, gamma: 0.5);
            Console.WriteLine("Starting policy");
            env.PrintPolicy(policy);
            Console.WriteLine();

            int runs = policy.EvaluateMC(1000);
            env.PrintVs(policy);
        }

        static void EvaluateStateActionValuesWithMC()
        {
            Environment env = new Environment();
            Console.WriteLine("Initial gridworld");
            env.Print();
            Console.WriteLine();

            // Monte carlo method with exploring stars
            Policy sap = new Policy(env, gamma: 0.5);
            Console.WriteLine("Starting SA policy");
            env.PrintPolicy(sap);
            Console.WriteLine();

            for (int i = 0; i < 10; i++)
            {
                sap.IterateMC(100);
                env.PrintPolicy(sap);
            }
            env.PrintsSAV(sap);
        }

        static void EvaluateOnPolicy()
        {
            Environment env = new Environment();
            Console.WriteLine("Initial gridworld");
            env.Print();
            Console.WriteLine();

            Policy p = new Policy(env, gamma: 0.5);
            Console.WriteLine("Starting policy");
            env.PrintPolicy(p);
            Console.WriteLine();

            p.OnPolicyFirstVisitMCControl(limit: 10000, epsilon: 0.2);
            env.PrintPolicy(p);
            env.PrintsSAV(p);
        }

        static void EvaluateOffPolicy()
        {
            Environment env = new Environment();
            //Console.WriteLine("Initial gridworld");
            //env.Print();
            //Console.WriteLine();

            // target policy
            Policy p = new Policy(env, gamma: 0.5);
            // behaviour policy
            Policy mu = new Policy(env);
            Console.WriteLine("Starting policies");
            env.PrintPolicy(p);
            //env.PrintPolicy(mu);
            Console.WriteLine();

            p.OffPolicyEveryVisitMCControl(mu, limit: 10000);
            env.PrintPolicy(p);
            env.PrintsSAV(p);
        }

        static void Main(string[] args)
        {
            Console.WriteLine("1 - Evaluate policy.");
            Console.WriteLine("2 - Iterate State-Action pair values with exploring starts.");
            Console.WriteLine("3 - Iterate on-policy first visit MC w/soft-policy");
            Console.WriteLine("4 - Iterate off-policy every visit MC");
            string decision = Console.ReadLine();
            switch (decision)
            {
                case "1":
                    EvaluatePolicyWithMC();
                    break;
                case "2":
                    EvaluateStateActionValuesWithMC();
                    break;
                case "3":
                    EvaluateOnPolicy();
                    break;
                case "4":
                    EvaluateOffPolicy();
                    break;
                default:
                    break;
            }

        }
    }
}
