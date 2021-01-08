# lpscheduler

This is a simple Python solver for "who-when-what" scheduling problems using linear programming. Given sets `I`, `J`, and `K`, respectively, of people (who), times (when), and jobs (what), such problems consist of finding an assignment `I x J -> K` subject to various constraints and preferences. Common constraints include job capacities limiting the number of people on a given job at the same time, or per-person requirements specifying a minimum (or maximum) number of certain job assignments. If all constraints and preferences are linear, then the natural structure of the problem motivates a representation as an integer linear program (ILP) with binary indicator variables `x[i,j,k]`, where `x[i,j,k] = 1` if person `i` at time `j` is assigned to job `k` and zero otherwise. The structural constraints are then `sum(x[i,j,:]) = 1` for each `(i, j)`. The resulting ILP is solved using [CVXOPT](http://cvxopt.org) with a [GLPK](https://www.gnu.org/software/glpk/) backend.

In the more realistic setting that nonlinear constraints exist so that an ILP model is inadequate, the core solver here can still be used for the linear subproblems that arise during the course of solving the full nonlinear problem. Indeed, a next step in making this a truly useful tool would be to wrap the current solver in an interactive workflow to assist a human-driven solution process.

## Overview

The main purpose of lpscheduler is to provide a convenient interface for model building by translating human-readable identifiers (i.e., names) into internal array indices for GLPK. The basic constructs are linear `Expression`s associating each variable with a coefficient; these can be scaled and added together to form constraints and cost functions. The `Scheduler` keeps track of all such quantities and, on calling `solve`, assembles the corresponding standard-form ILP then dispatches to GLPK. On output, the solution is returned as a table over the "who-when" axes.

Some additional functionalities are provided for sampling from the solution space when the solution is not unique:

- `setrandcost`: set a random cost function (for pure-constraint problems)
- `shuffle`: permute internal variable indices

There is really not much sophistication here, so it's perhaps best now just to demonstrate with a simple example.

## Demo

The following example solves a simple chore assignment problem:

```python
from lpscheduler import *

# set up problem space
who  = ['Barack', 'Hillary', 'Donald', 'Justin']
when = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
what = ['dishes', 'laundry', 'vacuum', 'free']

# initialize scheduler
s = Scheduler(who, when, what)

# constraint: every chore done each day
for _what in what:
    for _when in when:
        expr = sum(s(_who, _when, _what) for _who in who)
        s.addcons(Constraint(expr, '>=', 1))

# constraint: everyone does each chore at least once
for _who in who:
    for _what in what:
        expr = sum(s(_who, _when, _what) for _when in when)
        s.addcons(Constraint(expr, '>=', 1))

# constraint: Barack needs Mondays off
cons = Constraint(s('Barack', 'Mon', 'free'), '=', 1)
s.addcons(cons)

# constraint: Hillary must do laundry on either Tue or Thu
expr = sum(s('Hillary', _when, 'laundry') for _when in ['Tue', 'Thu'])
s.addcons(Constraint(expr, '=', 1))

# constraint: Donald does dishes only once
cons = Constraint(sum(s('Donald', _when, 'dishes') for _when in when), '<=', 1)
s.addcons(cons)

# constraint: Justin can't vacuum on MWF
for _when in ['Mon', 'Wed', 'Fri']:
    cons = Constraint(s('Justin', _when, 'vacuum'), '=', 0)
    s.addcons(cons)

# randomize cost to sample solution space
s.setrandcost()

# run solver and show solution
options = {'LPX_K_MSGLEV': 0}  # suppress GLPK output
t = s.solve(**options)
print(s.format(t))
```

with sample output:

```
        |   Mon     Tue     Wed     Thu     Fri
--------+----------------------------------------
 Barack |   free  laundry  vacuum  dishes laundry
Hillary |  dishes  dishes   free  laundry  vacuum
 Donald |  vacuum  vacuum laundry   free   dishes
 Justin | laundry   free   dishes  vacuum   free
```
