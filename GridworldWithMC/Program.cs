using System;
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
                return Tuple.Create(newState, 1.00);
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
        public double? Value { get; set; }
        // state return
        public double? G { get; set; }
        // number of times state has been visited(first time)
        public int N { get; set; }
        // list of actions
        public Dictionary<Action, StateActionPolicy> Actions { get; }
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
        public double? Value { get; set; }
        // state action return
        public double? G { get; set; }
        // number of times state action has been visited(first time)
        public int N { get; set; }
        // marks action as used by Policy
        public bool InPolicy { get { return P > 0; } }
        // probability of activation in policy
        public double P { get; set; }

        public StateActionPolicy()
        {
            Value = null;
        }
    }

    class Policy
    {
        Dictionary<State, PolicyState> P;
        private Environment env;
        private double gamma;
        private Random rnd;

        public double V(State s)
        {
            return P[s].Value.GetValueOrDefault(double.MinValue);
        }

        public double V(State s, Action a)
        {
            return P[s].Actions[a].Value.GetValueOrDefault(double.MinValue);
        }

        public Dictionary<Action, StateActionPolicy> A(State s)
        {
            return P[s].Actions;
        }

        private IEnumerable<Tuple<State, Action, double>> GenerateWalk(State s, Action a)
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
                a = P[s].NextAction;
            }
        }

        /// <summary>
        /// Generates an episode starting from S0, A0, following π
        /// </summary>
        /// <returns></returns>
        private List<Tuple<State, Action, double>> GenerateEpisode(bool exploringStart = true)
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
                a = P[s].NextAction;
            }
            // Generate a walk from that position
            foreach (var item in GenerateWalk(s, a))
            {
                episode.Add(item);
            } 
            return episode;
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
                var episode = GenerateEpisode();
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
                    if (p.G.HasValue)
                    {
                        if (p.Value.HasValue)
                        {
                            p.Value += (p.G.Value - p.Value) / (double)++p.N;
                        }
                        else
                        {
                            p.Value = p.G.Value;
                            p.N = 1;
                        }
                        p.G = null;
                    }
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
                var episode = GenerateEpisode();

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
                    if (sap.G.HasValue)
                    {
                        if (sap.Value.HasValue)
                        {
                            sap.Value += (sap.G.Value - sap.Value) / (double)++sap.N;
                        }
                        else
                        {
                            sap.Value = sap.G.Value;
                            sap.N = 1;
                        }
                        sap.G = null;
                    }
                }
            }
            foreach (var s in env.States)
            {
                // control
                // greedily mark only max yielding actions
                double? max = P[s].Actions.Max(kvp => kvp.Value.Value);
                double size = P[s].Actions.Where(kvp => kvp.Value.Value == max).Count();
                foreach (var item in P[s].Actions)
                {
                    item.Value.P = item.Value.Value == max ? 1 / size : 0;
                }
            }
            return limit;
        }

        public int OnPolicyFirstVisitMCControl(int limit = 1000, double epsilon = 0.1)
        {
            for (int i = 0; i < limit; i++)
            {
                // Generate an episode
                var episode = GenerateEpisode(exploringStart:false);

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
                    if (sap.G.HasValue)
                    {
                        if (sap.Value.HasValue)
                        {
                            sap.Value += (sap.G.Value - sap.Value) / (double)++sap.N;
                        }
                        else
                        {
                            sap.Value = sap.G.Value;
                            sap.N = 1;
                        }
                        sap.G = null;
                    }

                    // control
                    // softly mark only max yielding actions
                    double? max = P[s].Actions.Max(kvp => kvp.Value.Value);
                    double size = P[s].Actions.Where(kvp => kvp.Value.Value == max).Count();
                    double x = (1 - epsilon) / size + epsilon / (double)P[s].Actions.Count;
                    foreach (var item in P[s].Actions)
                    {
                        item.Value.P = item.Value.Value == max ? x : epsilon / (double)P[s].Actions.Count;
                    }
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

        static void Main(string[] args)
        {
            Console.WriteLine("1 - Evaluate policy.");
            Console.WriteLine("2 - Iterate State-Action pair values with exploring starts.");
            Console.WriteLine("3 - Iterate on-policy first visit MC w/soft-policy");
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
                default:
                    break;
            }

        }
    }
}
