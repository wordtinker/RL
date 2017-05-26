using System;
using System.Collections.Generic;
using System.Linq;

namespace MonteCarloTreeSearch
{
    class PolicyState
    {
        // the exploration parameter
        private double c = 0.01;
        private Random rnd;
        // number of times state has been visited
        public int N { get; set; }
        // list of actions
        public Dictionary<Action, StateActionPolicy> Actions { get; }
        public void Rebalance()
        {
            // using upper confidence bound as value
            foreach (StateActionPolicy p in Actions.Values)
            {
                if (p.N == 0)
                {
                    p.Value = double.PositiveInfinity;
                }
                else
                {
                    p.Value = p.W / p.N + c * Math.Sqrt(Math.Log(N) / p.N);
                }
            }

            // Update probabilities of calling action
            double max = Actions.Values.Max(p => p.Value);
            double size = Actions.Where(kvp => kvp.Value.Value == max).Count();
            // policy is greedy
            foreach (var item in Actions)
            {
                item.Value.P = item.Value.Value == max ? 1 / size : 0;
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
        // number of times state action has been visited
        public int N { get; set; }
        // number of winning times
        public int W { get; set; }
        // marks action as used by Policy
        public bool InPolicy { get { return P > 0; } }
        // probability of activation in policy
        public double P { get; set; }
        public StateActionPolicy()
        {
            Value = 0;
        }
    }

    interface IPolicy
    {
        void AddState(State s);
        bool ContainsState(State s);
        Action GetAction(State s);
        void Update(State s, Action a, double r);
        Dictionary<Action, StateActionPolicy> A(State s);
    }

    class SoftPolicy : IPolicy
    {
        public Dictionary<State, PolicyState> p = new Dictionary<State, PolicyState>();

        public Dictionary<Action, StateActionPolicy> A(State s)
        {
            return p[s].Actions;
        }

        public void AddState(State s)
        {
            p[s] = new PolicyState(s);
        }

        public bool ContainsState(State s)
        {
            return p.ContainsKey(s);
        }

        public Action GetAction(State s)
        {
            return p[s].NextAction;
        }

        public void Update(State s, Action a, double r)
        {
            p[s].N++;
            p[s].Actions[a].N++;
            if (r > 0)
            {
                p[s].Actions[a].W++;

            }
            p[s].Rebalance();
        }
    }

    class RandomPolicy : IPolicy
    {
        private PolicyState policyState;

        public Dictionary<Action, StateActionPolicy> A(State s)
        {
            return policyState.Actions;
        }

        public void AddState(State s)
        {
            throw new NotImplementedException();
        }

        public bool ContainsState(State s)
        {
            return true;
        }

        public Action GetAction(State s)
        {
            return policyState.NextAction;
        }
        public void Update(State s, Action a, double r)
        {
            throw new NotImplementedException();
        }

        public RandomPolicy()
        {
            State dummy = new State(-1, -1, StateType.NonTerminal);
            policyState = new PolicyState(dummy);
        }
    }

    class MCTS
    {
        private AEnvironment env;
        private State startState;
        private IPolicy policy;
        private IPolicy simulator;
        private Stack<Tuple<State, Action>> episode;
        private double? Selection()
        {
            State s = startState;
            while (policy.ContainsState(s))
            {
                Action a = policy.GetAction(s);
                var sr = env.GetReward(s, a);
                episode.Push(Tuple.Create(s, a));
                s = sr.Item1;
                if (s.Type == StateType.Terminal) return sr.Item2;
            }
            // Phase 2. Expand tree
            Expansion(s);
            return null;
        }
        private void Expansion(State s)
        {
            policy.AddState(s);
        }
        private double Simulation()
        {
            State s = episode.Peek().Item1;
            double r;
            do
            {
                Action a = simulator.GetAction(s);
                var sr = env.GetReward(s, a);
                s = sr.Item1;
                r = sr.Item2;
            } while (s.Type != StateType.Terminal);
            return r;
        }
        private void Update(double reward)
        {
            while (episode.Count > 0)
            {
                var sa = episode.Pop();
                policy.Update(sa.Item1, sa.Item2, reward);
            }
        }
        public void UpdatePolicy(int n)
        {
            for (int i = 0; i < n; i++)
            {
                double? finalReward = null;
                // Phase 1. Select moves while we have state in policy
                finalReward = Selection();
                // Phase 3. Simulate the tail of episode
                if (!finalReward.HasValue) finalReward = Simulation();
                // Phase 4. Back-propagation
                Update(finalReward.Value);    
            }
        }
        public MCTS(AEnvironment env, State startState, IPolicy policy, IPolicy simulator)
        {
            this.env = env;
            this.startState = startState;
            this.policy = policy;
            this.simulator = simulator;
            episode = new Stack<Tuple<State, Action>>();
            policy.AddState(startState);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. We dont know all the states of Env, and can't keep all states
            // indexer memory
            // 2. Hence, policy should very general and with gamma=1
            // 3. Task is to find an optimal way from start point 
            // to final state in the windy world.

            // Create an environment
            WindyGrid env = new WindyGrid();
            env.Print();
            State s = env.StartState;
            int n = 0;
            while (true)
            {
                // Create a policy for selection/expansion phase
                // deterministic
                IPolicy policy = new SoftPolicy();
                // Create a policy for simulation phase
                IPolicy randomMove = new RandomPolicy();
                MCTS search = new MCTS(new ProxyWind(env), s, policy, randomMove);
                search.UpdatePolicy(1000);

                env.PrintPolicy(policy);
                break;
                // choose action
                Action a = policy.GetAction(s);
                Console.WriteLine(a);
                s = env.GetReward(s, a).Item1;
                n++;
                if (s.Type == StateType.Terminal) break;
            }
            Console.WriteLine(n);
        }
    }
}
