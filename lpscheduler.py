"""
lpscheduler: a simple linear-programming-based scheduling utility
"""

import copy
import numpy as np
import operator

from cvxopt import glpk, matrix, spmatrix

"""
Sparse linear expression as a sum of terms of the form `coeff * x[index]`, where
`x` is the variable array and `index` is a (who, when, what) integer triplet.
Expressions can be scaled and added together. Internally represented by a
dictionary mapping each `index` to `coeff`.
"""
class Expression:
    def __init__(self, coeffs):
        self.coeffs = coeffs  # dict: index -> coeff

    def __str__(self):
        return 'Expression(%s)' % str(self.coeffs)

    # iterate over (index, coeff) pairs
    def __iter__(self):
        return ((index, coeff) for (index, coeff) in self.coeffs.iteritems())

    def __add__(self, other):
        expr = Expression(copy.deepcopy(self.coeffs))
        expr += other
        return expr

    def __iadd__(self, other):
        for (index, coeff) in other:
            if index in self.coeffs: self.coeffs[index] += coeff
            else:                    self.coeffs[index]  = coeff
        return self

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        self += -other
        return self

    def __neg__(self):
        self *= -1
        return self

    # scalar multiplication
    def __rmul__(self, a):
        expr = Expression(copy.deepcopy(self.coeffs))
        expr *= a
        return expr

    def __imul__(self, a):
        for index in self.coeffs: self.coeffs[index] *= a
        return self

    """
    Like __str__ but uses Scheduler instance to translate indices to human-
    readable (who, when, what) triplet.
    """
    def pretty_str(self, scheduler):
        def _pretty_str(index, coeff):
            who  = scheduler.who [index[0]]
            when = scheduler.when[index[1]]
            what = scheduler.what[index[2]]
            return '(%s, %s, %s): %s' % (who, when, what, coeff)
        s = ', '.join((_pretty_str(index, coeff) for (index, coeff) in self))
        return 'Expression({%s})' % s

"""
Version of `sum` that can act on Expressions.
"""
def xsum(sequence):
    return reduce(operator.add, sequence)

"""
Sparse linear constraint of the form `expr ctype cval`, where:
- `expr` : sparse linear expression
- `ctype`: constraint type ('=', '<=', or '>=')
- `val`  : constraint value
"""
class Constraint:
    def __init__(self, expr, ctype, val):
        if ctype not in ('=', '<=', '>='):
            raise ValueError('unsupported constraint type %s' % ctype)
        self.expr  = expr
        self.ctype = ctype
        self.val   = val

    def __str__(self):
        return 'Constraint(%s %s %s)' % (self.expr, self.ctype, self.val)

    # like Expression.pretty_str
    def pretty_str(self, scheduler):
        return 'Constraint(%s %s %s)' % (self.expr.pretty_str(scheduler),
                                         self.ctype, self.val)

"""
Scheduler for "who-when-what" assignment problems.

Variable space is set of binary variables `x[i,j,k]`, where `(i, j, k)` indexes
(who, when, what). A value of `x[i,j,k] = 1` means that person `i` at time `j`
is assigned to job `k`. Structural constraints: `sum(x[i,j,:]) = 1` for each
`(i, j)`.
"""
class Scheduler:
    """
    Initialize scheduler. Problem space defined by (who, when, what) arrays of
    human-readable identifiers, to be referenced by __call__.
    """
    def __init__(self, who, when, what):
        self.who  = np.array(who)
        self.when = np.array(when)
        self.what = np.array(what)
        self.whod  = dict(zip(self.who , xrange(len(self.who ))))
        self.whend = dict(zip(self.when, xrange(len(self.when))))
        self.whatd = dict(zip(self.what, xrange(len(self.what))))
        self.cost = None
        self.cons = []
        for who in self.who:
            for when in self.when:
                expr = xsum((self(who, when, what) for what in self.what))
                self.addcons(Constraint(expr, '=', 1))

    """
    Main interface for model building. Provides translation from
    (who, when, what) triplet to internal model index. Technically returns an
    Expression for that index with unit coefficient.
    """
    def __call__(self, who, when, what):
        i = self.whod [who ]
        j = self.whend[when]
        k = self.whatd[what]
        return Expression({(i,j,k): 1})

    """
    Add constraint.
    """
    def addcons(self, cons):
        self.cons.append(cons)

    """
    Set linear cost function (of type Expression).
    """
    def setcost(self, expr):
        self.cost = expr

    """
    Solve scheduling problem. Launches GLPK backend to solve corresponding ILP.
    Returns solution as "who-when" table with entries corresponding to assigned
    "what" indices. Can pass keyword arguments to control solver options.
    """
    def solve(self, **kwargs):
        # convert to standard form
        I = len(self.who )
        J = len(self.when)
        K = len(self.what)
        index1 = np.reshape(xrange(I*J*K), (I,J,K))
        N = index1.size
        c = matrix(np.zeros(N))
        if self.cost is not None:
            for (index, coeff) in self.cost: c[index1[index]] = coeff
        An, Ax, Ai, Aj, b = 0, [], [], [], []
        Gn, Gx, Gi, Gj, h = 0, [], [], [], []
        for cons in self.cons:
            if cons.ctype == '=':
                for (index, coeff) in cons.expr:
                    Ax.append(coeff)
                    Ai.append(An)
                    Aj.append(index1[index])
                b.append(cons.val)
                An += 1
            elif cons.ctype == '<=' or cons.ctype == '>=':
                s = 1 if cons.ctype == '<=' else -1
                for (index, coeff) in cons.expr:
                    Gx.append(s*coeff)
                    Gi.append(Gn)
                    Gj.append(index1[index])
                h.append(s*cons.val)
                Gn += 1
        A = spmatrix(Ax, Ai, Aj)
        b = matrix(b, tc='d')
        if Gn == 0:
            G = matrix(np.zeros((1,N)))
            h = matrix(0, tc='d')
        else:
            G = spmatrix(Gx, Gi, Gj, size=(Gn,N))
            h = matrix(h, tc='d')

        # solve with GLPK
        glpk.options.update(kwargs)
        status, x = glpk.ilp(c, G, h, A, b, B=set(xrange(N)))

        # process output
        if x is None:
            raise RuntimeError('no solution, exit status "%s"' % status)
        t = np.empty((I,J), dtype=int)
        X = np.reshape(x, index1.shape)
        for i in xrange(I):
            for j in xrange(J):
                what = np.nonzero(X[i,j,:])[0][0]
                t[i,j] = what
        return t

    """
    Show solution as human-readable formatted table.
    """
    def show(self, t):
        wholen  = max(len(who ) for who  in self.who )
        whenlen = max(len(when) for when in self.when)
        whatlen = max(len(what) for what in self.what)
        l = max(whenlen, whatlen)
        s = ([' '*wholen, '|'] + [when.center(l) for when in self.when])
        s = ' '.join(s)
        print s
        print ('-'*(wholen + 1) + '+' + '-'*len(s))[:len(s)]
        for (i, who) in enumerate(self.who):
            line = [who.center(wholen), '|']
            for (j, when) in enumerate(self.when):
                what = self.what[t[i,j]]
                line.append(what.center(l))
            print ' '.join(line)

    # like `show` but prints as CSV; useful for importing into other programs
    def tocsv(self, t):
        print ',' + ','.join(self.when)
        for (i, who) in enumerate(self.who):
            line = who
            for (j, when) in enumerate(self.when):
                what = self.what[t[i,j]]
                line += ',' + what
            print line

    """
    Set random cost function. Useful for sampling from solution space of pure-
    constraint problem.
    """
    def setrandcost(self):
        I = len(self.who )
        J = len(self.when)
        K = len(self.what)
        r = np.random.rand(I, J, K)
        self.cost = xsum([Expression({(i,j,k): r[i,j,k] for i in xrange(I)
                                                        for j in xrange(J)
                                                        for k in xrange(K)})])

    """
    Randomly shuffle indices. Useful for sampling from general solution space.
    """
    def shuffle(self):
        def _shuffle(expr, whop, whenp, whatp):
            coeffs = expr.coeffs
            _expr = {}
            for index in coeffs:
                i = whop [index[0]]
                j = whenp[index[1]]
                k = whatp[index[2]]
                _expr[(i,j,k)] = coeffs[index]
            return Expression(_expr)
        whop  = np.random.permutation(len(self.who ))
        whenp = np.random.permutation(len(self.when))
        whatp = np.random.permutation(len(self.what))
        self.who  = self.who [np.argsort(whop )]
        self.when = self.when[np.argsort(whenp)]
        self.what = self.what[np.argsort(whatp)]
        self.whod  = {key: whop [self.whod [key]] for key in self.whod }
        self.whend = {key: whenp[self.whend[key]] for key in self.whend}
        self.whatd = {key: whatp[self.whatd[key]] for key in self.whatd}
        if self.cost: self.cost = _shuffle(self.cost, whop, whenp, whatp)
        _cons = []
        for cons in self.cons:
            expr = _shuffle(cons.expr, whop, whenp, whatp)
            _cons.append(Constraint(expr, cons.ctype, cons.val))
        self.cons = _cons
