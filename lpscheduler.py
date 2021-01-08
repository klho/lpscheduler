"""
lpscheduler: a simple linear-programming-based scheduling utility
"""

import numpy as np
from copy import deepcopy
from cvxopt import glpk, matrix, spmatrix

class Expression:
    """
    Sparse linear expression as a sum of terms of the form `coeff * x[index]`,
    where `x` is the variable array and `index` is a (who, when, what) integer
    triplet. Expressions can be scaled and added together. Internally
    represented by a dictionary mapping each `index` to `coeff`.
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs  # dict: index -> coeff

    def __str__(self):
        return 'Expression({})'.format(self.coeffs)

    def __iter__(self):
        """Iterate over (index, coeff) pairs."""
        return ((index, coeff) for (index, coeff) in self.coeffs.items())

    def __add__(self, other):
        expr = deepcopy(self)
        expr += other
        return expr

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        if other == 0: return self  # for use with `sum`
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

    def __mul__(self, a):
        """Scalar multiplication."""
        return a * self

    def __rmul__(self, a):
        expr = deepcopy(self)
        expr *= a
        return expr

    def __imul__(self, a):
        for index in self.coeffs: self.coeffs[index] *= a
        return self

    def pretty_str(self, scheduler):
        """
        Like __str__ but uses Scheduler instance to translate indices to human-
        readable (who, when, what) triplets.
        """
        def _pretty_str(index, coeff):
            who  = scheduler.who [index[0]]
            when = scheduler.when[index[1]]
            what = scheduler.what[index[2]]
            return '({}, {}, {}): {}'.format(who, when, what, coeff)
        s = ', '.join((_pretty_str(index, coeff) for (index, coeff) in self))
        return 'Expression({})'.format(s)

class Constraint:
    """
    Sparse linear constraint of the form `expr ctype cval`, where:
    - `expr` : sparse linear expression
    - `ctype`: constraint type ('=', '<=', or '>=')
    - `val`  : constraint value
    """
    def __init__(self, expr, ctype, val):
        if ctype not in ('=', '<=', '>='):
            raise ValueError('unsupported constraint type {}'.format(ctype))
        self.expr  = expr
        self.ctype = ctype
        self.val   = val

    def __str__(self):
        return 'Constraint({} {} {})'.format(self.expr, self.ctype, self.val)

    def pretty_str(self, scheduler):
        """Like Expression.pretty_str for human-readable formatting."""
        return 'Constraint({} {} {})'.format(self.expr.pretty_str(scheduler),
                                             self.ctype, self.val)

class Scheduler:
    """
    Scheduler for "who-when-what" assignment problems.

    Variable space is set of binary variables `x[i,j,k]`, where `(i, j, k)`
    indexes (who, when, what). A value of `x[i,j,k] = 1` means that person `i`
    at time `j` is assigned to job `k`. Structural constraints:
    `sum(x[i,j,:]) = 1` for each `(i, j)`.
    """

    def __init__(self, who, when, what):
        """
        Initialize scheduler. Problem space defined by (who, when, what) arrays
        of human-readable identifiers, to be referenced by __call__.
        """
        self.who  = np.array(who )
        self.when = np.array(when)
        self.what = np.array(what)
        self.whod  = dict(zip(self.who , range(len(self.who ))))
        self.whend = dict(zip(self.when, range(len(self.when))))
        self.whatd = dict(zip(self.what, range(len(self.what))))
        self.cost = None
        self.cons = []
        for who in self.who:
            for when in self.when:
                expr = sum((self(who, when, what) for what in self.what))
                self.addcons(Constraint(expr, '=', 1))

    def __call__(self, who, when, what):
        """
        Main interface for model building. Provides translation from
        (who, when, what) triplet to internal model index. Technically returns
        an Expression for that index with unit coefficient.
        """
        i = self.whod [who ]
        j = self.whend[when]
        k = self.whatd[what]
        return Expression({(i,j,k): 1})

    def addcons(self, cons):
        """Add constraint."""
        self.cons.append(cons)

    def setcost(self, expr):
        """Set linear cost function (of type Expression)."""
        self.cost = expr

    def solve(self, **kwargs):
        """
        Solve scheduling problem. Launches GLPK backend to solve corresponding
        ILP. Returns solution as "who-when" table with entries corresponding to
        assigned "what" indices. Can pass keyword arguments to control solver
        options.
        """
        # convert to standard form
        I = len(self.who )
        J = len(self.when)
        K = len(self.what)
        index1 = np.reshape(range(I*J*K), (I,J,K))
        N = index1.size
        c = matrix(np.zeros(N))
        if self.cost is not None:
            for (index, coeff) in self.cost: c[int(index1[index])] = coeff
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
        A = spmatrix(np.array(Ax), np.array(Ai), np.array(Aj))
        b = matrix(b, tc='d')
        if Gn == 0:
            G = matrix(np.zeros((1,N)))
            h = matrix(0, tc='d')
        else:
            G = spmatrix(np.array(Gx), np.array(Gi), np.array(Gj), size=(Gn,N))
            h = matrix(h, tc='d')

        # solve with GLPK
        glpk.options.update(kwargs)
        status, x = glpk.ilp(c, G, h, A, b, B=set(range(N)))

        # process output
        if x is None:
            raise RuntimeError('no solution, exit status "{}"'.format(status))
        t = np.empty((I,J), dtype=int)
        X = np.reshape(x, index1.shape)
        for i in range(I):
            for j in range(J):
                what = np.nonzero(X[i,j,:])[0][0]
                t[i,j] = what
        return t

    def format(self, t):
        """Show solution as human-readable formatted table."""
        wholen  = max(len(who ) for who  in self.who )
        whenlen = max(len(when) for when in self.when)
        whatlen = max(len(what) for what in self.what)
        l = max(whenlen, whatlen)
        header = ([' '*wholen, '|'] + [when.center(l) for when in self.when])
        header = ' '.join(header)
        s = header + '\n'
        s += ('-'*(wholen + 1) + '+' + '-'*len(header))[:len(header)]
        for (i, who) in enumerate(self.who):
            line = [who.center(wholen), '|']
            for (j, when) in enumerate(self.when):
                what = self.what[t[i,j]]
                line.append(what.center(l))
            s += '\n' + ' '.join(line)
        return s

    def formatcsv(self, t, sep=','):
        """
        Like `format` but shows as CSV (or any other separator); useful for
        importing into other programs.
        """
        s = sep + sep.join(self.when)
        for (i, who) in enumerate(self.who):
            line = who
            for (j, when) in enumerate(self.when):
                what = self.what[t[i,j]]
                line += sep + what
            s += '\n' + line
        return s

    def setrandcost(self):
        """
        Set random cost function. Useful for sampling from solution space of
        pure-constraint problem.
        """
        I = len(self.who )
        J = len(self.when)
        K = len(self.what)
        r = np.random.rand(I, J, K)
        self.cost = sum([Expression({(i,j,k): r[i,j,k] for i in range(I)
                                                       for j in range(J)
                                                       for k in range(K)})])

    def shuffle(self):
        """
        Randomly shuffle indices. Useful for sampling from general solution
        space.
        """
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
