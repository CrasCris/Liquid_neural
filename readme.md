# Neural ODE vs Liquid Neural Networks

A comparison repository for **Neural Ordinary Differential Equations (Neural ODEs)** and **Liquid Neural Networks (LNNs)** applied to **time series regression** tasks.

---

## Neural Ordinary Differential Equations (Neural ODE)

Introduced by Chen et al. (2018), Neural ODEs replace discrete residual layers with a continuous-time dynamics model. Instead of stacking layers, the hidden state evolves according to an ODE:

```
dh(t)/dt = f(h(t), t, θ)
```

where `h(t)` is the hidden state at time `t` and `f` is a neural network parameterized by `θ`.

**Forward pass:** An ODE solver (e.g., Runge-Kutta) integrates the dynamics from an initial state `h(t₀)` to a final state `h(T)`.

**Training:** Gradients are computed via the adjoint sensitivity method, which avoids backpropagating through solver steps and keeps memory usage constant with respect to depth.

### Key properties
- Continuous-depth model: equivalent to an infinitely deep residual network
- Adaptive computation: the ODE solver uses more steps where the dynamics are complex
- Memory efficient: O(1) memory for backpropagation via adjoint method
- Naturally handles irregularly-sampled time series by querying any time point

### Limitations
- Slower training due to ODE solver overhead
- Expressiveness constrained by the homeomorphism property (the learned flow must be topology-preserving)
- Can struggle with sharp discontinuities or highly stiff systems

---

## Liquid Neural Networks (LNN)

Introduced by Hasani et al. (2021) from MIT CSAIL, Liquid Neural Networks are a class of continuous-time recurrent neural networks inspired by the nervous system of *C. elegans*. Their hidden state dynamics are governed by:

```
dx(t)/dt = -[1/τ + f(x(t), I(t), θ)] · x(t) + f(x(t), I(t), θ)
```

where `x(t)` is the neuron state, `τ` is a time constant, and `I(t)` is the external input signal. The key insight is the **liquid** (input-dependent) time constant: the effective decay rate of each neuron adapts based on the current input, making the network causally liquid.

A closed-form approximation (Liquid Time-Constant networks, LTC) enables efficient training:

```
x(t + Δt) ≈ σ(A·x(t) + B·I(t)) / (1 + Δt · σ(A·x(t) + B·I(t)))
```

### Key properties
- Adaptive time constants: neurons respond differently depending on input context
- High expressiveness with very few parameters (small network footprint)
- Strong causal and temporal reasoning capabilities
- Robust and interpretable dynamics derived from biological inspiration
- Handles irregularly-sampled sequences natively due to continuous-time formulation

### Limitations
- Stiff ODE system can be challenging to solve numerically
- Closed-form approximations (LTC) trade off some accuracy for speed
- Less established ecosystem compared to standard RNN/Transformer variants

---

## Comparison for Time Series Regression

| Property | Neural ODE | Liquid Neural Network |
|---|---|---|
| **Dynamics** | Autonomous: `dh/dt = f(h, t, θ)` | Input-driven: `dx/dt = g(x, I, t, θ)` |
| **Time constants** | Fixed (implicit in `f`) | Adaptive, input-dependent |
| **Parameter efficiency** | Moderate | High (few params, strong expressiveness) |
| **Irregular sampling** | Native via ODE solver | Native via continuous-time formulation |
| **Memory (training)** | O(1) via adjoint | Depends on solver; LTC approximation is efficient |
| **Interpretability** | Low | Moderate (biologically inspired structure) |
| **Suitability for regression** | Good for smooth trends | Strong for complex, causal temporal patterns |
| **Solver overhead** | High (general ODE solver) | Lower with LTC closed-form |

### Task: Time Series Regression
Both models are evaluated on their ability to predict continuous target values from sequential input data. The comparison focuses on:

- **Accuracy**: MSE / MAE on held-out test sequences
- **Sample efficiency**: performance as a function of training set size
- **Computational cost**: training time and inference latency
- **Generalization**: behavior on sequences with irregular time steps or distribution shift

---

## References

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS 2018.
- Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). *Liquid Time-constant Networks*. AAAI 2021.
- Lechner, M., & Hasani, R. (2020). *Learning Long-Term Dependencies in Irregularly-Sampled Time Series*. NeurIPS 2020.
