import random
import matplotlib.pyplot as plt
import numpy as np

##### Input Parameters #######

# names of species
all_names = ['U235', 'Th231', 'Pa231', 'Ac227',
             'Th227', 'Fr223', 'Ra223', 'Rn219', 
             'Po215', 'Pb211', 'Bi211', 'Ti207',
             'Po211', 'Pb207']
# initial populations
N = np.array([20000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# information about possible reactions
all_reactions = [
    # from, to, number of reactants, number produced, rate of reaction in 1/seconds*1e15
    ['U235', 'Th231', 1, 1, 1/2.22097645e16 * 1e15],
    ['Th231', 'Pa231', 1, 1, 1/91872 * 1e15], 
    ['Pa231', 'Ac227', 1, 1, 1/1.03380489e12 * 1e15],
    ['Ac227', 'Th227', 1, 1, 1/687088949 * 1e15],
    ['Th227', 'Ra223', 1, 1, 1/1617235.2 * 1e15],
    ['Ac227', 'Fr223', 1, 1, 1/687088949 * 1e15],
    ['Fr223', 'Ra223', 1, 1, 1/1308 * 1e15],
    ['Ra223', 'Rn219', 1, 1, 1/987897.6 * 1e15],
    ['Rn219', 'Po215', 1, 1, 1/3.96 * 1e15],
    ['Po215', 'Pb211', 1, 1, 1/0.001778 * 1e15],
    ['Pb211', 'Bi211', 1, 1, 1/2166 * 1e15],

    ['Bi211', 'Ti207', 1, 1, 1/127.8 * 1e15],
    ['Ti207', 'Pb207', 1, 1, 1/286.2 * 1e15],

    ['Bi211', 'Po211', 1, 1, 1/127.8 * 1e15],
    ['Po211', 'Pb207', 1, 1, 1/0.516 * 1e15],
]
# run simulation until t >= T
# or until there are no more reactions left to be done
t = 0.0
T = 1000

##### Formatting Etc #######

# double-check all names are unique
assert all(sorted(all_names) == np.unique(all_names))
# make dict associating all names with an index
idx = {k: v for v, k in enumerate(all_names)}
# store reaction info in a more easily accessible way
reactions_dict = {}
for r,p,Nr,Np,_ in all_reactions:
    reactions_dict[(idx[r], idx[p])] = (Nr,Np)
# make rate array
rates = np.zeros((len(all_names), len(all_names)))
for reactant, product, N_r, N_p, rate in all_reactions:
    assert product in all_names and reactant in all_names
    rates[idx[reactant],idx[product]] = rate
# Initialize results list
data = []
data.append((t, *N))

##### Gillespie! #######

while t < T:
    # calculate probabilities of all possible events
    w = rates * np.array([[n]*len(N) for n in N])
    # total weights
    r = np.sum(w)
    if r == 0:
        break
    # time step
    dt = -np.log(random.uniform(0.0, 1.0)) / r
    t += dt
    # event probabilities
    P = w/r
    # then pick which event should happen
    event = 0
    probsum = np.sum(P.flatten()[:event])
    u = random.uniform(0.0, 1.0)
    while probsum < u:
        event += 1
        probsum = np.sum(P.flatten()[:event])
    # after this loop, 'event' is the index of the event chose
    r = (event-1) // len(N)
    p = (event-1) % len(N)
    # find the corresponding reaction
    Nr, Np = reactions_dict[(r,p)]
    # ensure we won't get negative numebers, ignore if we would
    if N[r] - Nr >=0:
        N[r] -= Nr
        N[p] += Np
    # record data
    data.append((t, *N))
    print(list(N), end='\r')

# plot
t_list = [d[0] for d in data]
for i in range(1, len(data[0])):
    N_list = [d[i] for d in data]
    plt.plot(t_list, N_list, label=all_names[i-1])
plt.legend(loc='best')
plt.show()
