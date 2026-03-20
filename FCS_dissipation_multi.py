import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
import multiprocessing
from functools import partial
import time
from tqdm import tqdm

# --- [1] Mathematical & Helper Functions ---

def T_matrix(theta): 
    return np.array(([[1-theta,theta],[theta,1-theta]]))


def get_projectors_from_index(action_index, M):
    theta = action_index / (M - 1) * 2 * np.pi
    basis0 = np.array([np.cos(theta/2), np.sin(theta/2)])
    basis1 = np.array([np.sin(theta/2), -np.cos(theta/2)])
    return np.outer(basis0, basis0), np.outer(basis1, basis1)

def traj_prob(theta, history, M, sig0, sig1, nu=np.array((1/2, 1/2))):
    dist = nu
    T = T_matrix(theta)
    L_curr = len(history)
    
    for l in range(L_curr):
        action_projectors, obs = history[l]
        obs = int(obs)
        P0, P1 = action_projectors

        prob00 = np.real(np.trace(P0 @ sig0))
        prob01 = np.real(np.trace(P1 @ sig0))
        prob10 = np.real(np.trace(P0 @ sig1))
        prob11 = np.real(np.trace(P1 @ sig1))

        O_matrix = np.array([[prob00, prob01], [prob10, prob11]])

        if l != L_curr - 1:
            O_diag = np.diag(O_matrix[:, obs])
            dist = T @ O_diag @ dist
        elif l == L_curr - 1:
            output = np.dot(O_matrix[:, obs], dist)

    return -np.log(output + 1e-9) # Added epsilon for numerical stability

def total_objective(theta, history_list, M, sig0, sig1):
    total_cost = 0
    for h in history_list:
        total_cost += traj_prob(theta, h, M, sig0, sig1)
    return total_cost



# --- [2] Simulation Components (Belief, Updates, Work) ---

def generate(L, p): 
    state = np.random.choice([0,1], p=[1/2, 1/2])
    pattern = []
    for l in range(L):
        pattern.append(state)
        # Flip probability depends on current state
        if state == 0: state = np.random.choice([0,1], p=[1-p, p])
        else: state = np.random.choice([0,1], p=[p, 1-p])
    return pattern

def belief(n):
    arr = np.zeros((n,2))
    eps = np.linspace(-0.4999,0.4999,n)
    for i in range(n):
        arr[i,:] = np.array([1/2 - eps[i], 1/2 + eps[i]], dtype=np.float32)
    return arr

def group(beliefs, eta):
    # Vectorized distance calculation for speed
    diffs = np.sum((beliefs - eta)**2, axis=1)
    return np.argmin(diffs)

def update_general(eta, P0, P1, p, sig0, sig1, outcome):
    proj = P0 if outcome == 0 else P1
    p0 = np.real(np.trace(proj @ sig0))
    p1 = np.real(np.trace(proj @ sig1))
    norm = p0 * eta[0] + p1 * eta[1]
    if norm == 0: norm = 1e-9
    new_eta = np.array((p0 * eta[0]/norm, p1 * eta[1]/norm))
    return new_eta @ T_matrix(p)

def work_extract_general(beliefs, index, state, P0, P1):
    exp_state = beliefs[index][0]*sig0 + beliefs[index][1]*sig1
    
    # Calculate Work
    lam_0 = np.real(np.trace(P0 @ exp_state))
    lam_1 = np.real(np.trace(P1 @ exp_state))
    w_0 = np.log(2) + np.log(lam_0 + 1e-9)
    w_1 = np.log(2) + np.log(lam_1 + 1e-9)
    work_val = np.array((w_0, w_1))
    
    # Calculate Outcome Prob
    prob_outcome = np.array((np.real(np.trace(P0 @ state)), np.real(np.trace(P1 @ state))))
    prob_outcome /= np.sum(prob_outcome) # Ensure normalization
    
    return work_val, prob_outcome



# --- [3] Policies (Opt & Helstrom) ---


def mapping(N, M, p, sig0, sig1):
    eta = belief(N) # Get the discretized belief grid
    mapping_dict = dict()

    for i in range(N): # Iterate through all current belief states
        current_eta = eta[i] 
        
        for j in range(M): # Iterate through all discrete actions
            
            # --- ADAPTER STEP: Convert Index -> Projectors ---
            # We need this to communicate with update_general
            P0, P1 = get_projectors_from_index(j, M)
            
            for k in range(2): # Iterate through outcomes (0 or 1)
                
                # 1. Calculate the Probability of this outcome (P_outcome)
                # We need to re-calculate this explicitly because 'update_general' 
                # uses it internally for normalization but doesn't return it.
                proj = P0 if k == 0 else P1
                
                # Likelihoods: P(outcome|sig0) and P(outcome|sig1)
                lik_0 = np.real(np.trace(proj @ sig0))
                lik_1 = np.real(np.trace(proj @ sig1))
                
                # Total Probability: P(k) = P(k|0)P(0) + P(k|1)P(1)
                prob_outcome = lik_0 * current_eta[0] + lik_1 * current_eta[1]
                
                # 2. Get the Next Belief State
                # Using the new unified update function
                transitted_eta = update_general(current_eta, P0, P1, p, sig0, sig1, k)
                
                # 3. Discretize: Find nearest neighbor in our grid
                index_of_new_belief = group(eta, transitted_eta)

                # 4. Store in Dictionary
                mapping_dict[i, j, k] = index_of_new_belief, prob_outcome

    return mapping_dict


def opt_policy(theta, r, N, M, L, sig0, sig1):
    """
    Optimizes the measurement policy using Dynamic Programming.
    """
    V = dict()
    eta = belief(N) # Get the discrete belief grid
    
    # 1. Pre-compute the transition map using the new general mapping function
    Map = mapping(N, M, theta, sig0, sig1) 

    # 2. Dynamic Programming Loop
    for t in range(L): 
        for i in range(N): # Iterate through all belief states
            
            # Store the expected value for every possible action j
            action_values = np.zeros(M) 
            
            # Current Expected State based on belief i: rho = p0*sig0 + p1*sig1
            prior_0, prior_1 = eta[i]
            rho_expected = prior_0 * sig0 + prior_1 * sig1
            
            for j in range(M): # Iterate through all discrete actions
                
                # --- ADAPTER: Convert Index -> Projectors ---
                P0, P1 = get_projectors_from_index(j, M)
                
                # --- CALCULATE IMMEDIATE REWARDS (WORK) ---
                # Work extracted depends on the measurement result and the EXPECTED state
                # Formula: w = log(2) + log(Trace(P @ rho_expected))
                
                lam_0 = np.real(np.trace(P0 @ rho_expected))
                lam_1 = np.real(np.trace(P1 @ rho_expected))
                
                # Numerical stability clip
                lam_0 = max(lam_0, 1e-12)
                lam_1 = max(lam_1, 1e-12)
                
                reward_0 = np.log(2) + np.log(lam_0)
                reward_1 = np.log(2) + np.log(lam_1)
                
                # --- EXPECTATION OVER OUTCOMES ---
                exp_value_action = 0 
                
                for k in range(2): # Iterate outcomes (0 or 1)
                    # Retrieve next state and prob from the pre-computed Map
                    index_next_belief, prob_k = Map[i, j, k]
                    
                    # Immediate reward for this outcome
                    r_k = reward_0 if k == 0 else reward_1
                    
                    # Future value (Value of the next belief state at the previous time step)
                    if t == 0:
                        future_val = 0 # Terminal condition (Value is 0 at end)
                    else:
                        # V[t-1] stores the optimal value for the previous stage
                        future_val = V[t-1, index_next_belief][1] 
                        
                    # Bellman Expectation: P(k) * (Immediate_Reward + Future_Value)
                    exp_value_action += prob_k * (r_k + future_val)

                action_values[j] = exp_value_action

            # 3. Optimization: Choose the action maximizing expected value
            best_J = np.max(action_values)
            best_A = np.argmax(action_values)
            
            # Store (Best Action Index, Best Value)
            V[t, i] = best_A, best_J
            
    # Return the Value dictionary and the value of the central belief state at the final step
    return V, V[L-1, int((N-1)/2)]


def forward_run_opt(sig0, sig1, p, N, M, L, policy, string):
    beliefs = belief(N)
    batt = 0
    hist = []
    eta = np.array((1/2, 1/2))
    index = group(beliefs, eta)

    for i in range(L):
        state = sig0 if string[i] == 0 else sig1
        
        # Policy Choice
        best_act = int(policy[L - (i+1),index][0])
        
        P0, P1 = get_projectors_from_index(best_act, M)
        
        # Work & Outcome
        work_vals, probs = work_extract_general(beliefs, index, state, P0, P1)
        outcome = np.random.choice(2, p=probs)
        
        batt += work_vals[outcome]
        hist.append([(P0, P1), outcome])
        
        # Update
        eta = update_general(eta, P0, P1, p, sig0, sig1, outcome)
        index = group(beliefs, eta)
        
    return batt, hist

# --- [4] The Worker Function for Multiprocessing ---

def run_single_repetition(rep_id, K, L, N, M, p, r, sig0, sig1, TOFE):
    """
    This function runs one full simulation for K steps.
    """
    # Re-seed random number generator for each process to ensure independence
    np.random.seed(int(time.time()) + rep_id)
    
    dissipation_run = np.zeros(K)
    histories = []
    theta = 0.5 # Initial guess
    #sqrt_K_threshold = int(np.sqrt(K))

    for k in range(K):
        # 1. Generate State Sequence
        FCS = generate(L, p)
        
        # 2. Choose Policy & Run
        policy, _ = opt_policy(theta, r, N, M, L, sig0, sig1)
        work, hist = forward_run_opt(sig0, sig1, theta, N, M, L, policy, FCS)
        
        # 3. Store Data
        histories.append(hist)
        dissipation_run[k] = TOFE - work
        
        # 4. Update Theta (Optimization)
        # Bounds and method must be robust
        res = minimize_scalar(total_objective, args=(histories, M, sig0, sig1), 
                              bounds=(0.001, 0.999), method='bounded')
        theta = res.x
        if k%100 == 0:
            print(k,"iteration",theta, flush=True)
    return dissipation_run

# --- [5] Main Execution Block ---

if __name__ == '__main__':
    # Parameters
    L, N, M = 5, 51, 11
    K = 5000
    R = 200  
    p = 0.9
    r = 0.2
    sig0 = np.array([[1,0],[0,0]])
    sig1 = np.array([[r,np.sqrt(r*(1-r))],[np.sqrt(r*(1-r)),1-r]])
    
    #TOFE
    TOFE = (opt_policy(p,r,N,M,L,sig0,sig1)[1])[1]
    print(TOFE)
    
    print(f"Starting Simulation with K={K}, R={R} using Multiprocessing...")
    
    worker = partial(run_single_repetition, K=K, L=L, N=N, M=M, p=p, r=r, sig0=sig0, sig1=sig1, TOFE=TOFE)
    
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores.")
    
    results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap to get results as they finish
        # tqdm wraps the iterator to create the progress bar
        for res in tqdm(pool.imap(worker, range(R)), total=R, desc="Simulating"):
            results.append(res)
        
    dissipation_matrix = np.vstack(results)
    np.save(f"dissipation_data_without_exploration_L{L}_M{M}_K{K}_R{R}.npy".format(L,N,M,K,R), dissipation_matrix)
    print("Simulation Complete. ")
    