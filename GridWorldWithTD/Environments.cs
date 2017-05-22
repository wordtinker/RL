using System;
using System.Collections.Generic;
using System.Linq;

namespace GridWorldWithTD
{

    public enum Action
    {
        Up, Down, Left, Right
    }

    public enum StateType
    {
        Terminal, NonTerminal
    }

    public class State
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

    abstract class AEnvironment
    {
        public abstract IEnumerable<State> StatesPlus { get; }
        public abstract IEnumerable<State> States { get; }
        public abstract Tuple<State, double> GetReward(State state, Action action);
        public abstract void Print();
        public abstract void PrintVs(Policy policy);
        public abstract void PrintsSAV(Policy policy);
        public abstract void PrintPolicy(Policy policy);
    }

    abstract class Grid : AEnvironment
    {
        protected State[][] states;

        public Grid(int x, int y)
        {
            states = new State[x][];
            for (int i = 0; i < x; i++)
            {
                states[i] = new State[y];
                for (int j = 0; j < y; j++)
                {
                    states[i][j] = new State(i, j);
                }
            }
        }

        public override IEnumerable<State> StatesPlus
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
        public override IEnumerable<State> States
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
        public override void Print()
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
        public override void PrintVs(Policy policy)
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
        public override void PrintsSAV(Policy policy)
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
        public override void PrintPolicy(Policy policy)
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
    }

    /// <summary>
    /// A standard gridworld,
    /// with start and goal states, but with one difference: there is a crosswind upward
    /// through the middle of the grid.The actions are the standard four—up, down,
    /// right, and left—but in the middle region the resultant next states are shifted upward
    /// by a “wind,” the strength of which varies from column to column.
    /// </summary>
    class WindyGrid : Grid
    {
        /// <summary>
        /// Hidden information.
        /// </summary>
        /// <param name="state"></param>
        /// <param name="action"></param>
        /// <returns></returns>
        private State NextState(State state, Action action)
        {
            int nexti;
            int nextj;
            switch (action)
            {
                // probability of every action is 100%
                case Action.Up:
                    nexti = state.i - 1;
                    nextj = state.j;
                    break;
                case Action.Down:
                    nexti = state.i + 1;
                    nextj = state.j;
                    break;
                case Action.Left:
                    nexti = state.i;
                    nextj = state.j - 1;
                    break;
                case Action.Right:
                    nexti = state.i;
                    nextj = state.j + 1;
                    break;
                default:
                    nexti = state.i;
                    nextj = state.j;
                    break;
            }

            // apply wind
            if (nextj > 2 && nextj < 9) nexti--;
            if (nextj > 5 && nextj < 8) nexti--;

            // fix out of bounds error
            if (nexti < 0) nexti = 0;
            if (nexti > 6) nexti = 6;
            if (nextj < 0) nextj = 0;
            if (nextj > 9) nextj = 9;

            return states[nexti][nextj];
        }
        /// <summary>
        /// Do not have to know next state, not like in DP solution.
        /// </summary>
        /// <param name="state"></param>
        /// <param name="action"></param>
        /// <returns>New stat and reward </returns>
        public override Tuple<State, double> GetReward(State state, Action action)
        {
            State newState = NextState(state, action);
            if (newState.Type == StateType.Terminal)
            {
                return Tuple.Create(newState, 3.00);
            }
            return Tuple.Create(newState, -1.00);
        }
        public WindyGrid() : base(7, 10)
        {
            states[3][7].Type = StateType.Terminal;
        }
    }

    /// <summary>
    /// Simple grid with two exits.
    /// </summary>
    class GridWorld : Grid
    {
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
        public override Tuple<State, double> GetReward(State state, Action action)
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

        public GridWorld() : base(4, 4)
        {
            states[0][0].Type = StateType.Terminal;
            states[3][3].Type = StateType.Terminal;
        }
    }
}
