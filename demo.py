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
s.show(t)
