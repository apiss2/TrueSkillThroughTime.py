# -*- coding: utf-8 -*-
"""
   Trueskill Through Time
   ~~~~~~~~~
   :copyright: (c) 2019-2020 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""
import math
import timeit
from numba import njit, types, typed
#import ipdb
import trueskill as ts

"""
TODO:
    Optimize.
    - Numba have several problems with jitclass
    - c++ may be a better solution
"""

BETA = 1.0
MU = 0.0
SIGMA = BETA * 6
GAMMA = BETA * 0.05
P_DRAW = 0.0
EPSILON = 1e-6
ITERATIONS = 10
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
inf = math.inf
PI = SIGMA**-2
TAU = PI * MU

class Environment(object):
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, gamma=GAMMA, p_draw=P_DRAW, epsilon=EPSILON, iterations=ITERATIONS):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.gamma = gamma
        self.p_draw = p_draw
        self.epsilon = epsilon
        self.iterations = iterations

@njit(types.f8(types.f8))
def erfc(x):
    #"""(http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    a = -0.82215223 + t * 0.17087277; b =  1.48851587 + t * a
    c = -1.13520398 + t * b; d =  0.27886807 + t * c; e = -0.18628806 + t * d
    f =  0.09678418 + t * e; g =  0.37409196 + t * f; h =  1.00002368 + t * g
    r = t * math.exp(-z * z - 1.26551223 + t * h)
    return r if not(x<0) else 2.0 - r 

#timeit.timeit(lambda: erfc(0.9) , number=10000)/10000

@njit(types.f8(types.f8))
def erfcinv(y):
    if y >= 2: return -inf
    if y < 0: raise ValueError('argument must be nonnegative')
    if y == 0: return inf
    if not (y < 1): y = 2 - y
    t = math.sqrt(-2 * math.log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in [0,1,2]:
        err = erfc(x) - y
        x += err / (1.12837916709551257 * math.exp(-(x**2)) - x * err)
    return x if (y < 1) else -x

#timeit.timeit(lambda: erfcinv(0.9) , number=10000)/10000

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8))
def tau_pi(mu,sigma):
    if sigma > 0.0:
        pi_ = sigma ** -2
        tau_ = pi_ * mu
    elif (sigma + 1e-5) < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        pi_ = inf
        tau_ = inf
    return tau_, pi_

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8))
def mu_sigma(tau_,pi_):
    if pi_ > 0.0:
        sigma = math.sqrt(1/pi_)
        mu = tau_ / pi_
    elif pi_ + 1e-5 < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        sigma = inf 
        mu = 0.0
    return mu, sigma
        
#timeit.timeit(lambda: mu_sigma(1.0,2.0) , number=10000)/10000

@njit(types.f8(types.f8,types.f8,types.f8))
def cdf(x, mu=0, sigma=1):
    z = -(x - mu) / (sigma * sqrt2)
    return (0.5 * erfc(z))

@njit(types.f8(types.f8,types.f8,types.f8))
def pdf(x, mu, sigma):
    normalizer = (sqrt2pi * sigma)**-1
    functional = math.exp( -((x - mu)**2) / (2*sigma**2) ) 
    return normalizer * functional

@njit(types.f8(types.f8,types.f8,types.f8))
def ppf(p, mu, sigma):
    return mu - sigma * sqrt2  * erfcinv(2 * p)

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8,types.f8,types.f8))
def v_w(mu, sigma, margin,tie):
    if not tie:
        _alpha = (margin-mu)/sigma
        v = pdf(-_alpha,0,1) / cdf(-_alpha,0,1)
        w = v * (v + (-_alpha))
    else:
        _alpha = (-margin-mu)/sigma
        _beta  = ( margin-mu)/sigma
        v = (pdf(_alpha,0,1)-pdf(_beta,0,1))/(cdf(_beta,0,1)-cdf(_alpha,0,1))
        u = (_alpha*pdf(_alpha,0,1)-_beta*pdf(_beta,0,1))/(cdf(_beta,0,1)-cdf(_alpha,0,1))
        w =  - ( u - v**2 ) 
    return v, w

@njit(types.UniTuple(types.f8, 2)(types.f8,types.f8,types.f8,types.f8))
def trunc(mu, sigma, margin, tie):
    v, w = v_w(mu, sigma, margin, tie)
    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(1-w)
    return mu_trunc, sigma_trunc

def approx(N, margin, tie):
    mu, sigma = trunc(N.mu, N.sigma, margin, tie)
    return Gaussian(mu, sigma)

#timeit.timeit(lambda: trunc(1.0,2.0,0.0,False), number=10000)/10000



@njit(types.f8(types.f8,types.f8))
def compute_margin(p_draw, sd):
    return abs(ppf(0.5-p_draw/2, 0.0, sd ))

def max_tuple(t1, t2):
    return max(t1[0],t2[0]), max(t1[1],t2[1])

def gr_tuple(tup, threshold):
    return (tup[0] > threshold) or (tup[1] > threshold)

def podium(xs):
    return sortperm(xs)

def sortperm(xs):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(xs))]

def dict_diff(old, new):
    step = (0., 0.)
    for a in old:
        step = max_tuple(step, old[a].delta(new[a]))
    return step

class Gaussian(object):
    def __init__(self,mu=MU, sigma=SIGMA):
        if sigma >= 0.0:
            self.mu, self.sigma = mu, sigma
        else:
            raise ValueError(" sigma should be greater than 0 ")
    
    @property
    def tau(self):
        if self.sigma > 0.0:
            return self.mu * (self.sigma**-2)
        else:
            return inf
        
    @property
    def pi(self):
        if self.sigma > 0.0:
            return self.sigma**-2
        else:
            return inf
    
    def __iter__(self):
        return iter((self.mu, self.sigma))
    def __repr__(self):
        return 'N(mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)
    def __add__(self, M):
        return Gaussian(self.mu + M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __sub__(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __mul__(self, M):
        _tau, _pi = self.tau + M.tau, self.pi + M.pi
        mu, sigma = mu_sigma(_tau, _pi)
        return Gaussian(mu, sigma)        
    def __truediv__(self, M):
        _tau = self.tau - M.tau; _pi = self.pi - M.pi
        mu, sigma = mu_sigma(_tau, _pi)
        return Gaussian(mu, sigma)
    def forget(self,gamma,t):
        return Gaussian(self.mu, math.sqrt(self.sigma**2 + t*gamma**2))
    def delta(self, M):
        return abs(self.mu - M.mu) , abs(self.sigma - M.sigma) 
    def exclude(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 - M.sigma**2) )
    def isapprox(self, M, tol=1e-4):
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)
    
#timeit.timeit(lambda: Gaussian(1.0,2.0) , number=10000)/10000

N01 = Gaussian(0,1)
N00 = Gaussian(0,0)
Ninf = Gaussian(0,inf)
Nms = Gaussian(MU, SIGMA)

class Rating(object):
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, gamma=GAMMA, draw=Ninf):
        self.prior = Gaussian(mu,sigma)
        self.beta = beta
        self.gamma = gamma
        self.draw = draw
    def performance(self):
        return Gaussian(self.prior.mu, math.sqrt(self.prior.sigma**2 + self.beta**2))
    def __repr__(self):
        return 'Rating(mu=%.3f, sigma=%.3f)' % (self.prior.mu, self.prior.sigma) 

class team_variable(object):
    def __init__(self, prior=Ninf, likelihood_lose=Ninf, likelihood_win=Ninf, likelihood_draw=Ninf):
        self.prior = prior
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
        self.likelihood_draw = likelihood_draw
        
    @property
    def p(self):
        return self.prior*self.likelihood_lose*self.likelihood_win*self.likelihood_draw
    @property
    def posterior_win(self):
        return self.prior*self.likelihood_lose*self.likelihood_draw
    @property
    def posterior_lose(self):
        return self.prior*self.likelihood_win*self.likelihood_draw
    @property
    def likelihood(self):
        return self.likelihood_win*self.likelihood_lose*self.likelihood_draw

def performance(team):
    res = N00
    for rating in team:
        res += rating.performance()
    return res

class draw_messages(object):
    def __init__(self,prior = Ninf, prior_team = Ninf, likelihood_lose = Ninf, likelihood_win = Ninf):
        self.prior = prior
        self.prior_team = prior_team
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
    
    @property
    def p(self):
        return self.prior_team*self.likelihood_lose*self.likelihood_win
    
    @property
    def posterior_win(self):
        return self.prior_team*self.likelihood_lose
    
    @property
    def posterior_lose(self):
        return self.prior_team*self.likelihood_win
    
    @property
    def likelihood(self):
        return self.likelihood_win*self.likelihood_lose

class diff_messages(object):
    def __init__(self, prior=Ninf, likelihood=Ninf):
        self.prior = prior
        self.likelihood = likelihood
    @property
    def p(self):
        return self.prior*self.likelihood

class Game(object):
    def __init__(self, teams, result, p_draw=0.0):
        if len(teams) != len(result): raise ValueError("len(teams) != len(result)")
        if (0.0 > p_draw) or (1.0 <= p_draw): raise ValueError ("0.0 <= proba < 1.0")
        
        self.teams = teams
        self.result = result
        self.p_draw = p_draw
        self.likelihoods = []
        self.evidence = 0.0
        self.compute_likelihoods()
        
    def __len__(self):
        return len(self.result)
    
    def size(self):
        return [len(team) for team in self.teams]
    
    def performance(self,i):
        return performance(self.teams[i])    
    
    def graphical_model(self):
        g = self 
        r = g.result
        o = sortperm(r) 
        t = [team_variable(g.performance(o[e]),Ninf, Ninf, Ninf) for e in range(len(g))]
        d = [diff_messages(t[e].prior - t[e+1].prior, Ninf) for e in range(len(g)-1)]
        tie = [r[o[e]]==r[o[e+1]] for e in range(len(d))]
        margin = [0.0 if g.p_draw==0.0 else compute_margin(g.p_draw, math.sqrt( sum([a.beta**2 for a in g.teams[o[e]]]) + sum([a.beta**2 for a in g.teams[o[e+1]]]) )) for e in range(len(d))] 
        g.evidence = 1.0
        for e in range(len(d)):
            mu, sigma = d[e].prior.mu, d[e].prior.sigma
            g.evidence *= cdf(margin[e],mu,sigma)-cdf(-margin[e],mu,sigma) if tie[e] else 1-cdf(margin[e],mu,sigma)
        return o, t, d, tie, margin
    
    def likelihood_analitico(self):
        g = self
        o, t, d, tie, margin = g.graphical_model()
        d = d[0].prior
        mu_trunc, sigma_trunc =  trunc(d.mu, d.sigma, margin[0], tie[0])
        if d.sigma==sigma_trunc:
            delta_div = d.sigma**2*mu_trunc - sigma_trunc**2*d.mu
            theta_div_pow2 = inf
        else:
            delta_div = (d.sigma**2*mu_trunc - sigma_trunc**2*d.mu)/(d.sigma**2-sigma_trunc**2)
            theta_div_pow2 = (sigma_trunc**2*d.sigma**2)/(d.sigma**2 - sigma_trunc**2)
        res = []
        for i in range(len(t)):
            team = []
            for j in range(len(g.teams[o[i]])):
                mu = 0.0 if d.sigma==sigma_trunc else g.teams[o[i]][j].prior.mu + ( delta_div - d.mu)*(-1)**(i==1)
                sigma_analitico = math.sqrt(theta_div_pow2 + d.sigma**2
                                            - g.teams[o[i]][j].prior.sigma**2)
                team.append(Gaussian(mu,sigma_analitico))
            res.append(team)
        return (res[0],res[1]) if o[0]<o[1] else (res[1],res[0])
    
    def likelihood_teams(self):
        g = self 
        o, t, d, tie, margin = g.graphical_model()
        step = (inf, inf); i = 0 
        while gr_tuple(step,1e-6) and (i < 10):
            step = (0., 0.)
            for e in range(len(d)-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                d[e].likelihood = approx(d[e].prior,margin[e],tie[e])/d[e].prior
                likelihood_lose = t[e].posterior_win - d[e].likelihood
                step = max_tuple(step,t[e+1].likelihood_lose.delta(likelihood_lose))
                t[e+1].likelihood_lose = likelihood_lose
            for e in range(len(d)-1,0,-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                d[e].likelihood = approx(d[e].prior,margin[e],tie[e])/d[e].prior
                likelihood_win = t[e+1].posterior_lose + d[e].likelihood
                step = max_tuple(step,t[e].likelihood_win.delta(likelihood_win))
                t[e].likelihood_win = likelihood_win
            i += 1
        if len(d)==1:
            d[0].prior = t[0].posterior_win - t[1].posterior_lose
            d[0].likelihood = approx(d[0].prior,margin[0],tie[0])/d[0].prior
        t[0].likelihood_win = t[1].posterior_lose + d[0].likelihood
        t[-1].likelihood_lose = t[-2].posterior_win - d[-1].likelihood
        return [ t[o[e]].likelihood for e in range(len(t)) ] 
    
    def compute_likelihoods(self):
        if len(self.teams)>2:
            m_t_ft = self.likelihood_teams()
            self.likelihoods = [[ m_t_ft[e] - self.performance(e).exclude(self.teams[e][i].prior) for i in range(len(self.teams[e])) ] for e in range(len(self))]
        else:
            self.likelihoods = self.likelihood_analitico()            
        
    @property
    def posteriors(self):
        return [[ self.likelihoods[e][i] * self.teams[e][i].prior for i in range(len(self.teams[e]))] for e in range(len(self))]

#ta = [Rating(0,1),Rating(0,1),Rating(0,1)]
#tb = [Rating(0,1),Rating(0,1),Rating(0,1)]
#tc = [Rating(0,1),Rating(0,1),Rating(0,1)]
#td = [Rating(0,1),Rating(0,1),Rating(0,1)]
#time_tt = timeit.timeit(lambda: Game([ta,tb],[1,0]).posteriors, number=10000)/10000
#ta = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#tb = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#tc = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#td = [ts.Rating(0,1),ts.Rating(0,1),ts.Rating(0,1)]
#time_ts = timeit.timeit(lambda: ts.rate([ta,tb],[1,0]), number=10000)/10000
#time_ts/time_tt

class Skill(object):
    def __init__(self, forward=Ninf, backward=Ninf, likelihood=Ninf, elapsed=0):
        self.forward = forward
        self.backward = backward
        self.likelihood = likelihood
        self.elapsed = elapsed

class Agent(object):
    def __init__(self, rating, message, last_time):
        self.rating = rating
        self.message = message
        self.last_time = last_time
    
    def receive(self, elapsed):
        if self.message != Ninf:
            res = self.message.forget(self.rating.gamma, elapsed) 
        else:
            res = self.rating.prior
        return res

def clean(agents,last_time=False):
    for a in agents:
        agents[a].message = Ninf
        if last_time:
            agents[a].last_time = -inf

class Item(object):
    def __init__(self,name,likelihood):
        self.name = name
        self.likelihood = likelihood

class Team(object):
    def __init__(self, items, output):
        self.items = items
        self.output = output

class Event(object):
    def __init__(self, teams, evidence):
        self.teams = teams
        self.evidence = evidence
    def __repr__(self):
        return "Event({}, {})".format(self.names,self.result)
    @property
    def names(self):
        return [ [item.name for item in team.items] for team in self.teams]
    @property
    def result(self):
        return [ team.output for team in self.teams]

def get_composition(events):
    return [ [[ it.name for it in t.items] for t in e.teams] for e in events]

def get_results(events):
    return [ [t.output for t in e.teams ] for e in events]

def compute_elapsed(last_time, actual_time):
    return 0 if last_time == -inf  else ( 1 if last_time == inf else (actual_time - last_time))

class Batch(object):
    def __init__(self, composition, results, time, agents, env=Environment()):
        if len(composition)!= len(results): raise ValueError("len(composition)!= len(results)")
        
        this_agents = set( [a for teams in composition for team in teams for a in team ] )
        elapsed = dict([ (a,  compute_elapsed(agents[a].last_time, time) ) for a in this_agents ])
        
        self.skills = dict([ (a, Skill(agents[a].receive(elapsed[a]) ,Ninf ,Ninf , elapsed[a])) for a in this_agents  ])
        self.events = [Event([Team([Item(composition[e][t][a], Ninf) for a in range(len(composition[e][t])) ], results[e][t]  ) for t in range(len(composition[e])) ],0.0) for e in range(len(composition) )]
        self.time = time
        self.agents = agents
        self.iteration()
        
    
    def __repr__(self):
        return "Batch(time={}, events={})".format(self.time,self.events)
    def __len__(self):
        return len(self.events)
    def add_events(self, composition, results):
        b=self
        this_agents = set( [a for teams in composition for team in teams for a in team ] )
        for a in this_agents:
            elapsed = compute_elapsed(b.agents[a].last_time , b.time )  
            if a in b.skills:
                b.skills[a] = Skill(b.agents[a].receive(elapsed) ,Ninf ,Ninf , elapsed)
            else:
                b.skills[a].elapsed = elapsed
                b.skills[a].forward = b.agents[a].receive(elapsed)
        _from = len(b)+1
        for e in range(len(composition)):
            event = Event([Team([Item(composition[e][t][a], Ninf) for a in range(len(composition[e][t]))], results[e][t]) for t in range(len(composition[e])) ] , 0.0)
            b.events.append(event)
        b.iteration(_from)
    def posterior(self, agent):
        #TODO: esta funci'on debe pertenecer a la clase Skill 
        return self.skills[agent].likelihood*self.skills[agent].backward*self.skills[agent].forward
    def posteriors(self):
        res = dict()
        for a in self.skills:
            res[a] = self.posterior(a)
        return res
    def within_prior(self, item):
        r = self.agents[item.name].rating
        mu, sigma = self.posterior(item.name)/item.likelihood
        res = Rating(mu, sigma, r.beta, r.gamma)
        return res
    def within_priors(self, event):
        return [ [self.within_prior(item) for item in team.items ] for team in self.events[event].teams ]
    def iteration(self, _from=0):
        for e in range(_from,len(self)):
            teams = self.within_priors(e)
            result = self.events[e].result
            #TODO: El game tiene que recibir el margen
            g = Game(teams,result)
            for (t, team) in enumerate(self.events[e].teams):
                for (i, item) in enumerate(team.items):
                    self.skills[item.name].likelihood = (self.skills[item.name].likelihood / item.likelihood) * g.likelihoods[t][i]
                    item.likelihood = g.likelihoods[t][i]
            self.events[e].evidence = g.evidence
    def convergence(self, epsilon=1e-6, iterations = 20):
        step, i = (inf, inf), 0
        while gr_tuple(step, epsilon) and (i < iterations):
            #TODO: hay que copiar el posterior?
            old = self.posteriors().copy()
            self.iteration()
            step = dict_diff(old, self.posteriors())
            i += 1
        return i
    def forward_prior_out(self, agent):
        #TODO: Definir posterior_back en Skill
        return self.skills[agent].forward * self.skills[agent].likelihood
    def backward_prior_out(self, agent):
        N = self.skills[agent].likelihood*self.skills[agent].backward
        return N.forget(self.agents[agent].rating.gamma, self.skills[agent].elapsed) 
    def new_backward_info(self):
        for a in self.skills:
            self.skills[a].backward = self.agents[a].message
        return self.iteration()
    def new_forward_info(self):
        for a in self.skills:
            self.skills[a].forward = self.agents[a].receive(self.skills[a].elapsed) 
        return self.iteration()

#agents = dict()
#for k in ["a", "b", "c", "d", "e", "f"]:
    #agents[k] = Agent(Rating(25., 25.0/3, 25.0/6, 25.0/300 ) , Ninf, -inf)

#composition = [ [["a"],["b"]], [["c"],["d"]] , [["e"],["f"]] ]
#results = [[0,1],[1,0],[0,1]]
#batch = Batch(composition = composition, results = results, time = 0, agents = agents)
#timeit.timeit(lambda: Batch(composition = composition, results = results, time = 0, agents = agents), number=10000)/10000

class History(object):
    def __init__(self,composition,results,times=[],priors=dict(), env=Environment()):
        if len(composition) != len(results): raise ValueError("len(composition) != len(results)")
        if (len(times) > 0) and (len(composition) != len(times)): raise ValueError(" len(times) error ")
        
        self.size = len(composition)
        self.batches = []
        self.agents = dict([ (a, Agent(priors[a] if a in priors else Rating(env.mu, env.sigma, env.beta, env.gamma), Ninf, -inf)) for a in set( [a for teams in composition for team in teams for a in team] ) ])
        self.env = env
        #TODO: self.time podr'ia formar parte de Environment
        self.time = len(times)>0
        self.trueskill(composition,results,times)
        
    def __repr__(self):
        return "History(Events={}, Batches={}, Agents={})".format(self.size,len(self.batches),len(self.agents))
    def __len__(self):
        return self.size
    def trueskill(self, composition, results, times):
        o = sortperm(times) if len(times)>0 else [i for i in range(len(composition))]
        i = 0
        while i < len(self):
            #TODO: t tiene que ser i en caso de time
            #TODO: usar size y time 
            j, t = i+1, 1 if len(times) == 0 else times[o[i]]
            while (len(times)>0) and (j < len(self)) and (times[o[j]] == t): j += 1
            b = Batch([composition[k] for k in o[i:j]],[results[k] for k in o[i:j]], t, self.agents, self.env)        
            self.batches.append(b)
            for a in b.skills:
                self.agents[a].last_time = t if self.time else inf
                self.agents[a].message = b.forward_prior_out(a)
            i = j
    def iteration(self):
        step = (0., 0.)
        clean(self.agents)
        for j in reversed(range(len(self.batches)-1)):
            for a in self.batches[j+1].skills:
                self.agents[a].message = self.batches[j+1].backward_prior_out(a)
            old = self.batches[j].posteriors().copy()
            #TODO: evaluar que los bathces no tengan los agentes
            self.batches[j].new_backward_info()
            step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
        clean(self.agents)
        for j in range(1,len(self.batches)):
            for a in self.batches[j-1].skills:
                self.agents[a].message = self.batches[j-1].forward_prior_out(a)
            old = self.batches[j].posteriors().copy()
            self.batches[j].new_forward_info()
            step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
    
        if len(self.batches)==1:
            old = self.batches[0].posteriors().copy()
            self.batches[0].convergence()
            step = max_tuple(step, dict_diff(old, self.batches[0].posteriors()))
        
        return step
    def convergence(self, verbose=False):
        step = (inf, inf); i = 0
        while gr_tuple(step, self.env.epsilon) and (i < self.env.iterations):
            if verbose: print("Iteration = ", i, end=" ")
            step = self.iteration()
            i += 1
            if verbose: print(", step = ", step)
        if verbose: print("End")
        return step, i
    def learning_curves(self):
        res = dict()
        for b in self.batches:
            for a in b.skills:
                t_p = (b.time, b.posterior(a))
                if a in res:
                    res[a].append(t_p)
                else:
                    res[a] = [t_p]
        return res
    def log_evidence(self):
        return sum([math.log(event.evidence) for b in self.batches for event in b.events])

#composition = [ [["a"],["b"]], [["a"],["c"]] , [["b"],["c"]] ]
#results = [[0,1],[1,0],[0,1]]
#env = Environment(mu=0.0,sigma=6.0, beta=1.0, gamma=0.05, iterations=100)
#h = History(composition=composition, results=results, env=env)
#h.convergence(True)

#timeit.timeit(lambda: History(composition=composition, results=results, env=env), number=10000)/10000


#ta = [Rating(1.139,0.531,1.0,0.2125)]
#tb = [Rating(15.568,0.51,1.0,0.2125)]

#g = Game([ta,tb], [1,0], 0.0)
type(N00) == Gaussian 
timeit.timeit(lambda: isinstance(N00,Gaussian), number=10000)/10000
timeit.timeit(lambda: type(N00) == Gaussian, number=10000)/10000
timeit.timeit(lambda: Gaussian(0,0), number=10000)/10000



2.5e-6-1.8e-6
0.0000001
