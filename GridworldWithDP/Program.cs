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

        public List<Tuple<State, double>> NextState(State state, Action action)
        {
            List<Tuple<State, double>> actions = new List<Tuple<State, double>>();
            try
            {
                switch (action)
                {
                    // probability of every action is 100%
                    case Action.Up:
                        actions.Add(Tuple.Create(states[state.i - 1][state.j], 1.00));
                        break;
                    case Action.Down:
                        actions.Add(Tuple.Create(states[state.i + 1][state.j], 1.00));
                        break;
                    case Action.Left:
                        actions.Add(Tuple.Create(states[state.i][state.j - 1], 1.00));
                        break;
                    case Action.Right:
                        actions.Add(Tuple.Create(states[state.i][state.j + 1], 1.00));
                        break;
                    default:
                        actions.Add(Tuple.Create(state, 1.00));
                        break;
                }
            }
            catch (Exception)
            {
                actions.Add(Tuple.Create(state, 1.00));
            }
            return actions;
        }

        public double GetReward(State state, Action action, State newState)
        {
            if (newState.Type == StateType.Terminal)
            {
                return 1;
            }
            else if (state == newState)
            {
                return -2;
            }
            return -1;
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

        public void PrintPolicy(Policy policy)
        {
            foreach (var item in states)
            {
                string line1 = "";
                string line2 = "";
                string line3 = "";
                foreach (State s in item)
                {
                    if (s.Type == StateType.Terminal)
                    {
                        line1 += "xxx ";
                        line2 += "xxx ";
                        line3 += "xxx ";
                    }
                    else
                    {
                        Action[] actions = policy.A(s).ToArray();
                        line1 += actions.Contains(Action.Up) ? " ^  " : "    ";
                        line2 += actions.Contains(Action.Left) ? "< " : "  ";
                        line2 += actions.Contains(Action.Right) ? "> " : "  ";
                        line3 += actions.Contains(Action.Down) ? " v  " : "    ";
                    }
                }
                Console.WriteLine(line1);
                Console.WriteLine(line2);
                Console.WriteLine(line3);
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
        // state value
        public double Value { get; set; }
        public double NewValue { get; set; }
        // list of equiprobable actions
        public List<Action> Actions { get; set; }
        public PolicyState(State s)
        {
            Actions = new List<Action>();
            foreach (Action a in s.Actions)
            {
                Actions.Add(a);
            }
        }
    }

    class Policy
    {
        Dictionary<State, PolicyState> P;
        private Environment env;
        private double gamma;

        public double V(State s)
        {
            return P[s].Value;
        }

        public IEnumerable<Action> A(State s)
        {
            return P[s].Actions;
        }

        public int IterateValue(double theta = 0.01)
        {
            // initialize V(s) = 0 for all states
            foreach (PolicyState ps in P.Values)
            {
                ps.Value = 0;
            }

            int k = 0;
            double delta;
            // Iterate value until convergence
            do
            {
                delta = 0;
                k++;
                foreach (State s in env.States)
                {
                    double? bestValue = null;
                    Action bestAction;
                    // fetch all available actions from the environment
                    foreach (Action a in s.Actions)
                    {
                        double v = 0;
                        // look up the next state(s)
                        foreach (var nsp in env.NextState(s, a))
                        {
                            State ns = nsp.Item1;
                            double stateProb = nsp.Item2;
                            // get reward for transition triplet
                            double r = env.GetReward(s, a, ns);
                            v += stateProb * (r + gamma * V(ns));
                        }
                        // find best action among actions, only one even if there are many
                        if (bestValue == null || v >= bestValue)
                        {
                            bestValue = v;
                            bestAction = a;
                        }
                    }
                    delta = Math.Max(delta, Math.Abs(bestValue.Value - P[s].Value));
                    P[s].Value = bestValue.Value;
                }
            } while (delta >= theta);
            return k;
        }

        public int Evaluate(double theta = 0.01)
        {
            // initialize V(s) = 0 for all states
            foreach (PolicyState ps in P.Values)
            {
                ps.Value = 0;
            }

            int k = 0;
            double delta;
            // Evaluate policy until convergence
            do
            {
                delta = 0;
                k++;
                foreach (State s in env.StatesPlus)
                {
                    double v = 0;
                    // fetch all available actions from the policy
                    foreach (Action a in P[s].Actions)
                    {
                        // probability of taking action under current policy
                        double actionProb = 1 / (double)P[s].Actions.Count;
                        // look up the next state(s)
                        foreach (var nsp in env.NextState(s, a))
                        {
                            State ns = nsp.Item1;
                            double stateProb = nsp.Item2;
                            // get reward for transition triplet
                            double r = env.GetReward(s, a, ns);
                            v += actionProb * stateProb * (r + gamma * V(ns));
                        }
                    }
                    delta = Math.Max(delta, Math.Abs(v - P[s].Value));
                    P[s].NewValue = v;
                    // TODO in place update will converge too! most common way, a bit faster
                    //P[s].Value = v;
                }
                foreach (State s in P.Keys)
                {
                    P[s].Value = P[s].NewValue;
                }
            } while (delta >= theta);
            return k;
        }

        public bool Improve()
        {
            // TODO This algorithm has a subtle bug, in that it may never terminate if the policy continually switches
            // between two or more policies that are equally good. Check it.
            bool policyStable = true;
            foreach (State s in env.States)
            {
                List<Action> oldPolicy = P[s].Actions;
                Dictionary<Action, double> newPolicy = new Dictionary<Action, double>();
                foreach (Action a in s.Actions)
                {
                    double actionValue = 0;
                    // look up the next state(s)
                    foreach (var nsp in env.NextState(s, a))
                    {
                        State ns = nsp.Item1;
                        double stateProb = nsp.Item2;
                        // get reward for transition triplet
                        double r = env.GetReward(s, a, ns);
                        // calculate value of the action
                        actionValue += stateProb * (r + gamma * V(ns));
                    }
                    newPolicy[a] = actionValue;
                }
                // greedily take only max yielding actions
                double max = newPolicy.Max(kvp => kvp.Value);
                P[s].Actions = newPolicy.Where(kvp => kvp.Value == max).Select(kvp => kvp.Key).ToList();
                // check for stability
                if (!(P[s].Actions.Count == oldPolicy.Count && P[s].Actions.All(oldPolicy.Contains)))
                {
                    policyStable = false;
                }
            }
            return policyStable;
        }

        public Policy(Environment env, double gamma = 1)
        {
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
        static void Main(string[] args)
        {
            Environment env = new Environment();
            Console.WriteLine("Initial gridworld");
            env.Print();
            Console.WriteLine();

            Policy policy = new Policy(env, gamma: 1);
            Console.WriteLine("Starting policy");
            env.PrintPolicy(policy);
            Console.WriteLine();

            // Policy iteration process
            bool stable = false;
            do
            {
                // 1. Policy evaluation
                int runs = policy.Evaluate(theta: 0.5);
                Console.WriteLine($"Policy has been evaluated {runs} times.");
                env.PrintVs(policy);
                Console.WriteLine();
                // 2. Policy improvement
                stable = policy.Improve();
                Console.WriteLine($"Policy is stable: {stable}");
                env.PrintPolicy(policy);
                Console.WriteLine();
            } while (!stable);

            // Value iteration process
            Policy otherPolicy = new Policy(env, gamma: 0.5);
            Console.WriteLine("Starting policy");
            env.PrintPolicy(otherPolicy);
            Console.WriteLine();

            int takes = otherPolicy.IterateValue(theta: 0.01);
            otherPolicy.Improve();
            Console.WriteLine($"Value has been evaluated {takes} times.");
            env.PrintVs(otherPolicy);
            env.PrintPolicy(otherPolicy);
        }
    }
}
