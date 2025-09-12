
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import jax
import jax.numpy as jnp

def GKMS_opt(C,Q,
             tau_in=50,tau_out=50,gamma=70, 
             r=10,max_iter=200,device='cpu',
              dtype=torch.float64, printCost=True, 
             returnFull=False, 
             initialization='Full',
             init_args = None,
             full_grad=True,
             tol=1e-9,
             min_iter = 25,
             max_inneriters_balanced= 500,
             max_inneriters_relaxed= 500,
             eps=1e-4,
             lambda_reg=0.1):
    
    C = C.to(device=device, dtype=dtype)
    Q = Q.to(device=device, dtype=dtype).clamp_min(1e-18)
    
    if lambda_reg is not None:
        Q = stabilize_Q_init(Q, device=device, dtype=dtype, lambda_factor = lambda_reg)
    
    one_r = torch.ones((r), device=device, dtype=dtype)
    
    g = (1/r)*one_r
    Lambda = torch.diag(1.0 / g)
    
    N1, N2 = C.shape
    
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    a = one_N1 / N1
    b = one_N2 / N2
    
    errs = []
    gamma_k = gamma
    
    # Initialize duals for warm-start across iterations
    errs = []
    dual_1Q, dual_2Q = None, None
    
    for k in range(max_iter):
        
        gradQ, gamma_k = compute_grad_Q(C, Q, Lambda, 
                                               gamma, device, dtype,
                                               full_grad=full_grad)
        Q, dual_1Q, dual_2Q = logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, g, gamma_k, max_iter = max_inneriters_relaxed, \
                     device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in, \
                            dual_1 = dual_1Q, dual_2 = dual_2Q)
        g = Q.sum(dim=0).clamp_min(1e-18)
        Lambda = torch.diag(1.0 / g)
        cost = torch.trace(( (Q.T @ C) @ Q) @ Lambda)
        errs.append(cost.cpu())
        
        if k >= max(2, min_iter):
            if torch.abs(errs[-1] - errs[-2]) <= tol:
                break
    
    if printCost:
        with torch.no_grad():
            plt.plot(range(len(errs)), errs)
            plt.xlabel('Iterations')
            plt.ylabel('OT-Cost')
            plt.show()
    
    if returnFull:
        P_Full = Q @ Lambda @ Q.T
        return P_Full, errs
    else:
        return Q, g, errs

def logSinkhorn(grad, a, b, gamma_k, max_iter = 50, \
             device='cpu', dtype=torch.float64, \
                balanced=True, unbalanced=False, \
                tau=None, tau2=None, \
                recenter_every=30, tol=1e-12, \
                squeeze=True,
               dual_1 = None, dual_2 = None):
    
    a, b = (a / a.sum()), (b / b.sum())
    
    log_a = torch.log(a)
    log_b = torch.log(b)
    
    n, m = a.size(0), b.size(0)

    if dual_1 is None and dual_2 is None:
        f_k = torch.zeros((n), device=device)
        g_k = torch.zeros((m), device=device)
    else:
        f_k = dual_1
        g_k = dual_2
    
    epsilon = gamma_k**-1
    
    if not balanced:
        ubc = (tau/(tau + epsilon ))
        if tau2 is not None:
            ubc2 = (tau2/(tau2 + epsilon ))
    
    for it in range(max_iter):
        f_prev = f_k.clone()
        g_prev = g_k.clone()
        if balanced and not unbalanced:
            # Balanced
            f_k = f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1))
            g_k = g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0))
        elif not balanced and unbalanced:
            # Unbalanced
            f_k = ubc*(f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc2*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )
        else:
            # Semi-relaxed
            f_k = (f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )
            
        if it % recenter_every == 0:
            # Recenter potentials; gauge invariant
            alpha = f_k.mean()
            f_k -= alpha
            g_k += alpha
        
        if max((f_k-f_prev).abs().max(), (g_k-g_prev).abs().max()) < tol:
            break
    P = torch.exp(Cost(f_k, g_k, grad, epsilon, device=device))
    return P, f_k, g_k

def Cost(f, g, Grad, epsilon, device='cpu', dtype=torch.float64):
    '''
    A matrix which is using for the broadcasted log-domain log-sum-exp trick-based updates.
    ------Parameters------
    f: torch.tensor (N1)
        First dual variable of semi-unbalanced Sinkhorn
    g: torch.tensor (N2)
        Second dual variable of semi-unbalanced Sinkhorn
    Grad: torch.tensor (N1 x N2)
        A collection of terms in our gradient for the update
    epsilon: float
        Entropic regularization for Sinkhorn
    device: 'str'
        Device tensors placed on
    '''
    return -( Grad - torch.outer(f, torch.ones(Grad.size(dim=1), device=device)) - torch.outer(torch.ones(Grad.size(dim=0), device=device), g) ) / epsilon

def sinkhorn_transport(C, a, b, eps=1e-2, max_iter=1000, device='cpu', dtype=torch.float64):
    # Balanced Sinkhorn for P = diag(u) K diag(v), K = exp(-C/eps)
    # (non log-domain!)
    K = torch.exp(-C.to(device=device, dtype=dtype) / eps)
    u = torch.ones_like(a, device=device, dtype=dtype)
    v = torch.ones_like(b, device=device, dtype=dtype)
    # Avoid divide-by-zero
    a = a.clamp_min(1e-18); b = b.clamp_min(1e-18)
    for _ in range(max_iter):
        u = a / (K @ v).clamp_min(1e-18)
        v = b / (K.t() @ u).clamp_min(1e-18)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P

def monge_permutation(C):
    # Optimal permutation via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(C)
    n = C.shape[0]
    P = jnp.zeros_like(C)
    P = P.at[row_ind, col_ind].set(1.0)
    return col_ind, P

def _kmeanspp_init(X, k, rng):
    n = X.shape[0]
    centers = jnp.empty((k, X.shape[1]), dtype=X.dtype)
    i0 = rng.integers(n)
    centers = centers.at[0].set(X[i0])
    d2 = jnp.sum((X - centers[0])**2, axis=1)
    for t in range(1, k):
        s = d2.sum()
        probs = d2/s if s > 0 else np.ones(n)/n
        it = rng.choice(n, p=probs)
        centers = centers.at[t].set(X[it])
        d2 = jnp.minimum(d2, jnp.sum((X - centers[t])**2, axis=1))
    return centers

def _lloyds_kmeans(X, k, max_iter=250, tol=1e-6, random_state=0):
    rng = np.random.default_rng(random_state)
    centers = _kmeanspp_init(X, k, rng)
    for _ in range(max_iter):
        d2 = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = jnp.argmin(d2, axis=1)
        new_centers = jnp.zeros_like(centers)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                new_centers = new_centers.at[j].set(X[rng.integers(X.shape[0])])
            else:
                new_centers = new_centers.at[j].set(pts.mean(axis=0))
        if jnp.linalg.norm(new_centers - centers) <= tol:
            centers = new_centers
            break
        centers = new_centers
    d2 = jnp.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    labels = jnp.argmin(d2, axis=1)
    return labels, centers

def stabilize(Q, R, floor=1e-12):
    return Q.clamp_min(floor), R.clamp_min(floor)

def compute_grad_Q(C, Q, Lambda, gamma, device, dtype=torch.float64, full_grad=True):
    r = Lambda.shape[0]
    one_r = torch.ones((r), device=device, dtype=dtype)
    One_rr = torch.outer(one_r, one_r).to(device)
    gradQ = Wasserstein_Grad(C, Q, Lambda, device, \
                   dtype=torch.float64, full_grad=full_grad)
    normalizer = gradQ.abs().max()
    gamma_k = gamma / normalizer
    return gradQ, gamma_k

def Wasserstein_Grad(C, Q, Lambda, device, \
                   dtype=torch.float64, full_grad=True):
    
    gradQ = (C @ Q) @ Lambda.T
    if full_grad:
        # rank-one perturbation
        N1 = Q.shape[0]
        one_N1 = torch.ones((N1), device=device, dtype=dtype)
        gQ = Q.T @ one_N1
        w1 = torch.diag( (gradQ.T @ Q) @ torch.diag(1/gQ) )
        gradQ -= torch.outer(one_N1, w1)
    
    return gradQ

def stabilize_Q_init(Q, rand_perturb = False, 
                     lambda_factor = 0.1, max_inneriters_balanced= 300, 
                     device='cpu', dtype=torch.float64):
    """
    Initial condition Q (e.g. from annotation, if doing a warm-start) will not optimize if one-hot.
                ---e.g. if most of Q_t is sparse/a clustering, logQ_t = - inf which is unstable!
    
    Perturb to ensure there is non-zero mass everywhere.
    """
    # Add a small random or trivial outer product perturbation to ensure stability of one-hot encoded Q
    N2, r2 = Q.shape[0], Q.shape[1]
    b, gQ = torch.sum(Q, axis = 1), torch.sum(Q, axis = 0)
    eps_Q = torch.outer(b, gQ).to(device).type(dtype)
    
    # Yield perturbation, return
    Q_init = ( 1 - lambda_factor ) * Q + lambda_factor * eps_Q
    
    return Q_init
