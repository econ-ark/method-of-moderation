# Key Equations: The Method of Moderation

This document presents the core mathematical results of the Method of Moderation.

---

## 1. The Consumer's Problem

### Utility Function (CRRA)

$$u(c) = \begin{cases}
\frac{c^{1-\rho}}{1-\rho} & \text{if } \rho \neq 1 \\
\log c & \text{if } \rho = 1
\end{cases}$$

### Bellman Equation (Normalized)

$$v(m) = \max_{c} \left\{ u(c) + \beta \mathbb{E}\left[\Psi^{1-\rho} v(m')\right] \right\}$$

subject to:
- $a = m - c$ (budget constraint)
- $m' = a R + \theta'$ (transition)

### Euler Equation

$$u'(c) = \beta R \mathbb{E}\left[\Psi^{-\rho} u'(c')\right]$$

$$c^{-\rho} = \beta R \mathbb{E}\left[\Psi^{-\rho} (c')^{-\rho}\right]$$

---

## 2. Analytical Bounds

### Optimist's Consumption Function

The consumer who assumes $\theta = 1$ forever:

$$c_{\text{opt}}(m) = \kappa_{\min} (m + h)$$

where:
- $\kappa_{\min} = 1 - \frac{(\beta R)^{1/\rho}}{R}$ is the minimum MPC
- $h = \frac{\Gamma}{R - \Gamma}$ is human wealth

### Pessimist's Consumption Function

The consumer who assumes $\theta = \theta_{\min}$ forever:

$$c_{\text{pes}}(m) = \kappa_{\min} (m - m_{\min}) = \kappa_{\min} \cdot m^e$$

where:
- $m_{\min} = -h_{\text{pes}}$ is the natural borrowing constraint
- $m^e = m - m_{\min}$ is excess market resources

### Key Inequality

For the realist consumer:

$$c_{\text{pes}}(m) < c_{\text{real}}(m) < c_{\text{opt}}(m)$$

This holds for all $m > m_{\min}$.

---

## 3. The Moderation Ratio

### Definition

$$\omega(m) = \frac{c_{\text{real}}(m) - c_{\text{pes}}(m)}{c_{\text{opt}}(m) - c_{\text{pes}}(m)}$$

### Properties

- $\omega \in (0, 1)$ strictly for all $m > m_{\min}$
- $\omega \to 0$ as $m \downarrow m_{\min}$ (maximum precaution)
- $\omega \to 1$ as $m \to \infty$ (no precautionary saving)

### Interpretation

- $\omega = 0$: Realist = Pessimist (all precautionary saving)
- $\omega = 1$: Realist = Optimist (no precautionary saving)
- $\omega = 0.5$: Realist halfway between bounds

---

## 4. The Logit Transformation

### Log Excess Resources

$$\mu = \log(m - m_{\min}) = \log(m^e)$$

Domain: $\mu \in (-\infty, \infty)$ for $m \in (m_{\min}, \infty)$

### Chi Function (Logit of Moderation)

$$\chi(\mu) = \log\left(\frac{\omega(\mu)}{1 - \omega(\mu)}\right) = \text{logit}(\omega)$$

### Key Property: Asymptotic Linearity

As $\mu \to \infty$:

$$\chi'(\mu) \to 0$$

This means $\chi$ becomes flat for large wealth, preventing extrapolation errors.

### Inverse Transformation

$$\omega = \frac{1}{1 + e^{-\chi}} = \text{expit}(\chi)$$

---

## 5. Reconstruction Formula

Given an interpolated $\hat{\chi}(\mu)$, reconstruct consumption:

$$\hat{c}(m) = c_{\text{pes}}(m) + \hat{\omega}(m) \cdot \left(c_{\text{opt}}(m) - c_{\text{pes}}(m)\right)$$

where:

$$\hat{\omega}(m) = \text{expit}(\hat{\chi}(\log(m - m_{\min})))$$

**Guarantee**: $c_{\text{pes}}(m) < \hat{c}(m) < c_{\text{opt}}(m)$ always holds.

---

## 6. Tighter Upper Bound (Extension)

### Maximum MPC Constraint

Theory shows the MPC is bounded:

$$\kappa_{\min} \leq c'(m) \leq \kappa_{\max}$$

where $\kappa_{\max} = 1 - \wp^{1/\rho} \cdot (\beta R)^{1/\rho} / R$

### Tighter Upper Bound Function

$$c_{\text{tight}}(m) = c_{\text{opt}}(m) - (\kappa_{\max} - \kappa_{\min}) \cdot m^e$$

This provides a tighter constraint than $c_{\text{opt}}$.

---

## 7. Value Function Bounds (Extension)

### Transformed Value Function

$$\mathfrak{v}(m) = ((1-\rho) v(m))^{1/(1-\rho)}$$

### Value Function Moderation

Similar technique applies:

$$\mathfrak{v}_{\text{pes}}(m) < \mathfrak{v}_{\text{real}}(m) < \mathfrak{v}_{\text{opt}}(m)$$

---

## 8. Cusp Point

### Definition

The cusp point $\check{m}$ is where $c_{\text{tight}}$ crosses $c_{\text{pes}}$:

$$c_{\text{tight}}(\check{m}) = c_{\text{pes}}(\check{m})$$

### Formula

$$\check{m} = m_{\min} + \frac{\kappa_{\min} h^e}{\kappa_{\max} - \kappa_{\min}}$$

### Significance

For $m < \check{m}$: The tighter bound is tighter than pessimist (impossible)
For $m > \check{m}$: The tighter bound provides additional constraint

---

## 9. Stochastic Returns (Extension)

When returns are risky ($R$ is stochastic):

### Modified Patience Factor

$$\tilde{\Phi} = \mathbb{E}\left[(\beta \tilde{R})^{1/\rho}\right]$$

### Modified Bounds

The same framework applies with adjusted bounds accounting for return risk.

---

## Summary: The Algorithm in Equations

1. **Compute bounds**: $c_{\text{opt}}(m) = \kappa_{\min}(m + h)$, $c_{\text{pes}}(m) = \kappa_{\min} m^e$

2. **Solve EGM**: Get $(m_i, c_i)$ pairs from Euler equation

3. **Transform**: $\mu_i = \log(m_i - m_{\min})$, $\omega_i = \frac{c_i - c_{\text{pes}}(m_i)}{c_{\text{opt}}(m_i) - c_{\text{pes}}(m_i)}$, $\chi_i = \log(\omega_i / (1-\omega_i))$

4. **Interpolate**: Create $\hat{\chi}(\mu)$ from $\{(\mu_i, \chi_i)\}$

5. **Evaluate**: $\hat{c}(m) = c_{\text{pes}}(m) + \text{expit}(\hat{\chi}(\log(m - m_{\min}))) \cdot (c_{\text{opt}}(m) - c_{\text{pes}}(m))$
