# Goal-Oriented (GO) Reinforcement Learning (RL) Transport Layer (TL)
The framework for deploying CPSs, collecting application and network traces, constructing an RL environment, and training an RL agent. Includes the constructed environment and the utilized data traces.

##  Model and Control Parameters:

| Parameter                              | Value      | Parameter                               | Value      |
|-----------------------------------------|------------|-----------------------------------------|------------|
| State/control dimension                | $1$        | Tag window for RL state $W$             | $10$       |
| State matrices ${A}_i$                 | $1.2$      | Throughput window for RL state $W_T$    | $100$      |
| LQG input matrices ${B}^{LQG}_i$       | $1$        | LQG cost params ${Q}_i, {R}_i$          | $1$        |
| Process noise covariance matrix        | $1$        | RL loss function                        | Smooth L1  |
| PID Input matrices ${B}_i^{PID}$       | $0.5$      | Optimizer                               | AdamW      |
| Proportional gain $K_{p,i}$            | $2.3$      | Differential gain $K_{d, i}$            | $0.0015$   |
| Integral Gain $K_{i,i}$                | $120$      | Activation function                     | ReLu       |
| $1$-st fully-connected in $\times$ out   | $32\times512$ | $2$-nd fully-connected in$\times$out | $512\times512$ |
| $3$-d fully-connected in $\times$ out    | $512 \times 2$ | Batch size                          | $128$      |
