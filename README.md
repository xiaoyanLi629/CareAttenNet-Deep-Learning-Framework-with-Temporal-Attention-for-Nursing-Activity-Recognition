# Deep Learning Models for Nursing Activity Recognition - PyTorch Implementation

## Project Overview

This project implements five advanced deep learning models for automatic nursing activity recognition based on sensor data. Using the SONAR nursing activity dataset, we classify 20 different nursing activities through time series analysis and multiple neural network architectures.

### Core Features
- 🧠 **5 Advanced Model Architectures**: Baseline CNN-LSTM, Correlation-Aware CNN, Attention LSTM, Feature-Selective Network, and HybridNet
- 📊 **Ablation Study**: Systematic analysis of each component's contribution to model performance
- 🔧 **Overfitting Prevention**: Label smoothing, weight decay, learning rate scheduling, gradient clipping
- 📈 **Comprehensive Evaluation**: Accuracy, F1-score, precision, recall, ROC curves, confusion matrices
- 📝 **Complete Logging**: Automatic saving of training process and experimental results

## Problem Formulation

### Mathematical Definition of Nursing Activity Recognition

Nursing activity recognition can be formalized as a multivariate time series classification task. Let the inertial sensor system generate a $D$-dimensional feature vector $\mathbf{x}_t \in \mathbb{R}^D$ at time $t$, where $D = 70$ represents multimodal features obtained from 5 body sensors.

**Problem Definition**: Given a time series observation window of length $T$, $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T] \in \mathbb{R}^{T \times D}$, learn a mapping function:

$$f: \mathbb{R}^{T \times D} \rightarrow \mathbb{R}^C$$

that maps the time series window to a probability distribution over $C$ nursing activity categories, where $C = 20$ represents different nursing activity types.

### Physical Structure of Sensor Data

The multimodal sensor data has a clear physical structure reflecting different aspects of human movement:

$$\mathbf{x}_t = \begin{bmatrix} 
\mathbf{q}_t^{(1)}, \mathbf{q}_t^{(2)}, ..., \mathbf{q}_t^{(S)} \\
\dot{\mathbf{q}}_t^{(1)}, \dot{\mathbf{q}}_t^{(2)}, ..., \dot{\mathbf{q}}_t^{(S)} \\
\mathbf{v}_t^{(1)}, \mathbf{v}_t^{(2)}, ..., \mathbf{v}_t^{(S)} \\
\mathbf{m}_t^{(1)}, \mathbf{m}_t^{(2)}, ..., \mathbf{m}_t^{(S)}
\end{bmatrix}$$

where $S = 5$ represents the number of sensors:

- $\mathbf{q}_t^{(s)} \in \mathbb{R}^4$: Quaternion orientation representation of sensor $s$
- $\dot{\mathbf{q}}_t^{(s)} \in \mathbb{R}^4$: Quaternion derivative (related to angular velocity)
- $\mathbf{v}_t^{(s)} \in \mathbb{R}^3$: Linear velocity and acceleration components
- $\mathbf{m}_t^{(s)} \in \mathbb{R}^3$: Three-axis magnetic field intensity measurements

### Temporal Windowing and Data Preprocessing

**Temporal Window Construction**: To preserve temporal dependencies, fixed-length non-overlapping windows are employed:

$$\mathbf{W}_i = \{\mathbf{x}_{(i-1) \cdot \tau + 1}, \mathbf{x}_{(i-1) \cdot \tau + 2}, ..., \mathbf{x}_{i \cdot \tau}\}$$

where $\tau = 20$ is the window size and $i$ is the window index.

**Standardization Processing**: To ensure numerical stability, Z-score normalization is applied to each feature dimension:

$$\tilde{\mathbf{x}}_t^{(d)} = \frac{\mathbf{x}_t^{(d)} - \mu^{(d)}}{\sigma^{(d)}}$$

where $\mu^{(d)}$ and $\sigma^{(d)}$ are the mean and standard deviation of the $d$-th feature dimension on the training set, respectively.

### Class Imbalance Problem

The nursing activity data exhibits significant class imbalance, with the imbalance ratio defined as:

$$\rho = \frac{\max_{c \in \{1,...,C\}} |\mathcal{D}_c|}{\min_{c \in \{1,...,C\}} |\mathcal{D}_c|}$$

where $|\mathcal{D}_c|$ represents the number of samples in class $c$. In the dataset, $\rho \approx 156.7$, necessitating the use of weighted loss functions:

$$\mathcal{L}_{weighted} = -\sum_{i=1}^N w_{y_i} \log p(y_i | \mathbf{X}_i)$$

where the weight $w_c = \frac{N}{C \cdot |\mathcal{D}_c|}$ is used to balance class contributions.

### Data Leakage Prevention Strategy

**Subject-level Splitting**: To prevent data leakage, strict subject-based (participant-based) data splitting is performed:

$$\mathcal{S} = \mathcal{S}_{train} \cup \mathcal{S}_{val} \cup \mathcal{S}_{test}, \quad \mathcal{S}_{train} \cap \mathcal{S}_{val} \cap \mathcal{S}_{test} = \emptyset$$

where $\mathcal{S}$ represents the set of all subjects, ensuring that any subject's data appears in only one subset.

**Temporal Independence**: Non-overlapping windows ensure temporal independence between samples:

$$\mathbf{W}_i \cap \mathbf{W}_j = \emptyset, \quad \forall i \neq j$$

### Optimization Objective and Loss Function

**Primary Objective**: Minimize the expected risk of prediction errors:

$$\mathcal{R}(f) = \mathbb{E}_{(\mathbf{X}, y) \sim \mathcal{D}} [\ell(f(\mathbf{X}), y)]$$

where $\ell$ is the loss function and $\mathcal{D}$ is the true data distribution.

**Practical Loss**: Cross-entropy loss combined with label smoothing:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c}^{smooth} \log \hat{y}_{i,c}$$

where the smoothed label is defined as:

$$y_{i,c}^{smooth} = (1-\alpha) y_{i,c} + \frac{\alpha}{C}$$

The smoothing parameter $\alpha = 0.1$ is used to improve generalization ability.

### Evaluation Metrics

**Primary Metric**: Multi-class classification accuracy

$$\text{Accuracy} = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{I}[\arg\max_c \hat{y}_{i,c} = y_i]$$

**Auxiliary Metrics**: Weighted F1-score, precision, recall, and class-level performance analysis.

## Dataset Information

### SONAR Nursing Activity Dataset
- **Data Source**: Sensor data from real nursing environments
- **Feature Dimension**: 70-dimensional sensor features (quaternions, velocity, magnetic field, etc.)
- **Activity Categories**: 20 nursing activities (changing clothes, bed bath, kitchen preparation, etc.)
- **Number of Subjects**: 13 participants
- **Temporal Window**: 20 timesteps, non-overlapping windows
- **Data Split**: Subject-based split (70% training, 15% validation, 15% test)

## Theoretical Foundations and Methodology

### Problem Formalization Definition

Given multivariate time series data $\mathbf{X} = \{x_1, x_2, ..., x_T\}$, where $x_t \in \mathbb{R}^d$ represents the $d$-dimensional sensor feature vector at timestep $t$, our goal is to learn a mapping function $f: \mathbb{R}^{T \times d} \rightarrow \mathbb{R}^C$ that maps the time series window to one of $C$ nursing activity categories.

### Data Representation and Preprocessing

#### Feature Space Decomposition
Based on the physical properties of sensors, we decompose the 70-dimensional feature vector into four semantic groups:

$$\mathbf{x}_t = [\mathbf{q}_t; \mathbf{\dot{q}}_t; \mathbf{v}_t; \mathbf{m}_t]$$

where:
- $\mathbf{q}_t \in \mathbb{R}^{12}$: Quaternion features (orientation information)
- $\mathbf{\dot{q}}_t \in \mathbb{R}^{12}$: Quaternion derivatives (rate of orientation change)
- $\mathbf{v}_t \in \mathbb{R}^{24}$: Velocity features (motion information)
- $\mathbf{m}_t \in \mathbb{R}^{22}$: Magnetic field features (direction information)

#### Temporal Window Construction
A non-overlapping sliding window strategy is employed, with each window containing $W=20$ consecutive timesteps:

$$\mathbf{X}^{(i)} = [x_{(i-1) \cdot W + 1}, x_{(i-1) \cdot W + 2}, ..., x_{i \cdot W}]$$

To ensure label consistency, only windows with completely identical activity labels are retained.

## Model Architecture Details

### 1. Baseline CNN-LSTM: Hybrid Spatiotemporal Feature Learning

#### Design Motivation
Nursing activities have distinct spatiotemporal dual characteristics: local temporal patterns (e.g., instantaneous features of hand movements) and global temporal dependencies (e.g., temporal evolution of complete activities). The CNN-LSTM architecture captures both characteristics through hierarchical feature extraction.

#### Mathematical Modeling

**1D Convolutional Layer**:
$$\mathbf{h}^{(1)}_t = \sigma(W_1 * \mathbf{x}_{t:t+k-1} + b_1)$$

where $*$ denotes 1D convolution operation, $k$ is the kernel size, and $\sigma$ is the activation function.

**LSTM Layer**:

$$
\mathbf{f}_t = \sigma_g(W_f \mathbf{h}^{(2)}_t + U_f \mathbf{h}_{t-1} + b_f)
$$

$$
\mathbf{i}_t = \sigma_g(W_i \mathbf{h}^{(2)}_t + U_i \mathbf{h}_{t-1} + b_i)
$$

$$
\mathbf{o}_t = \sigma_g(W_o \mathbf{h}^{(2)}_t + U_o \mathbf{h}_{t-1} + b_o)
$$

$$
\tilde{\mathbf{c}}_t = \sigma_h(W_c \mathbf{h}^{(2)}_t + U_c \mathbf{h}_{t-1} + b_c)
$$

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \sigma_h(\mathbf{c}_t)
$$

**Bidirectional LSTM**:

$$
\overrightarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{h}^{(2)}_t, \overrightarrow{\mathbf{h}}_{t-1})
$$

$$
\overleftarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{h}^{(2)}_t, \overleftarrow{\mathbf{h}}_{t+1})
$$

$$
\mathbf{h}_t^{\text{bi}} = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]
$$

#### Detailed Architecture
```
Architecture Components:
├── 1D Conv Layer (input_size=70, filters=64, kernel=3, stride=1)
│   ├── BatchNorm1d(64)
│   ├── ReLU activation
│   └── Dropout(0.3)
├── 1D Conv Layer (input_size=64, filters=128, kernel=3, stride=1)
│   ├── BatchNorm1d(128)
│   ├── ReLU activation
│   └── Dropout(0.3)
├── Bidirectional LSTM Layer (input_size=128, hidden_size=64)
│   └── Output dimension: 128 (64×2)
├── Dropout(0.5)
├── Fully Connected Layer (128 → num_classes)
└── Softmax activation
```

#### Key Innovations
1. **Hierarchical Feature Extraction**: CNN captures local temporal patterns, LSTM models long-term dependencies
2. **Bidirectional Context**: Bidirectional LSTM utilizes future and past information
3. **Regularization Strategy**: Batch normalization and Dropout prevent overfitting

#### Data Utilization Method
- Directly processes raw 70-dimensional features
- Temporal window size: 20
- Preserves original relationships between features

### 2. Correlation-Aware CNN: Physics-Constrained Feature Learning

#### Design Motivation
Traditional CNNs treat all features as homogeneous, ignoring the physical structure of sensor data. Different sensor groups have different physical meanings and correlation patterns. This architecture performs grouped processing based on sensor physical properties, learning intra-group correlations and inter-group interactions.

#### Theoretical Foundation
**Feature Grouping Hypothesis**: Assume there exists feature grouping $\mathcal{G} = \{G_1, G_2, G_3, G_4\}$, where features within each group have stronger correlations, and learnable interaction patterns exist between groups.

**Intra-group Correlation Modeling**:

For the k-th feature group, apply specialized convolutional kernels:

$$
\mathbf{x}^{(k)}_t \in \mathbb{R}^{d_k}
$$

$$
\mathbf{h}^{(k)}_t = \sigma(W^{(k)} * \mathbf{x}^{(k)}_{t:t+w-1} + b^{(k)})
$$

**Inter-group Correlation Learning**:

Define correlation function:

$$
\rho: \mathbb{R}^{d_i} \times \mathbb{R}^{d_j} \rightarrow \mathbb{R}^{d_{ij}}
$$

$$
\mathbf{c}_{ij} = \rho(\mathbf{h}^{(i)}, \mathbf{h}^{(j)}) = \frac{\mathbf{h}^{(i)} \odot \mathbf{h}^{(j)}}{\|\mathbf{h}^{(i)}\|_2 \|\mathbf{h}^{(j)}\|_2}
$$

**Feature Fusion**:

$$
\mathbf{h}_{\text{fused}} = \text{Concat}([\mathbf{h}^{(1)}, \mathbf{h}^{(2)}, \mathbf{h}^{(3)}, \mathbf{h}^{(4)}, \mathbf{c}_{12}, \mathbf{c}_{13}, ..., \mathbf{c}_{34}])
$$

#### Detailed Architecture
```
Group Definitions:
├── Quaternion Group (G₁): [0:12]   - Orientation quaternions w,x,y,z
├── Quaternion Derivative Group (G₂): [12:24] - Rate of orientation change
├── Velocity Group (G₃): [24:48]    - Three-axis velocity and acceleration
└── Magnetic Field Group (G₄): [48:70]    - Magnetometer data

Processing Flow for Each Group:
├── 1D Grouped Convolution (group_conv1d, filters=32, kernel=3)
├── BatchNorm1d + ReLU
├── Global Average Pooling (AdaptiveAvgPool1d)
└── Output: Feature representation for each group

Correlation Computation:
├── L2 Normalization: h⁽ⁱ⁾_norm = h⁽ⁱ⁾ / ||h⁽ⁱ⁾||₂
├── Element-wise Product: c_ij = h⁽ⁱ⁾_norm ⊙ h⁽ʲ⁾_norm
└── Correlation Weights: α_ij (learnable parameters)

Final Fusion:
├── Feature Concatenation: [h⁽¹⁾, h⁽²⁾, h⁽³⁾, h⁽⁴⁾, c₁₂, c₁₃, c₁₄, c₂₃, c₂₄, c₃₄]
├── Fully Connected Layer (input_dim: 4×32 + 6×32 = 320)
└── Classification Output
```

#### Mathematical Derivation

**Advantages of Grouped Convolution**:
Parameter reduction:
$$\text{Reduction} = 1 - \frac{\sum_{k=1}^{4} d_k \cdot f_k}{d \cdot f}$$

where $d_k$ is the number of features in the $k$-th group, $f_k$ is the corresponding number of filters.

**Theoretical Basis for Correlation Metric**:
Cosine similarity variant:
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos(\theta)$$

Element-wise product captures feature correspondence:
$$\mathbf{c} = \mathbf{u}_{\text{norm}} \odot \mathbf{v}_{\text{norm}}$$

#### Key Innovations
1. **Physics-Aware Grouping**: Feature grouping based on sensor physical meaning
2. **Explicit Correlation Modeling**: Capturing inter-group relationships through learnable correlation functions
3. **Parameter Efficiency**: Grouped convolution significantly reduces number of parameters
4. **Domain Knowledge Integration**: Encoding sensor domain knowledge into network structure

### 3. Attention LSTM: Adaptive Temporal Attention Mechanism

#### Design Motivation
Nursing activities have different temporal importance distributions, and traditional LSTM treats all timesteps equally. This architecture dynamically identifies critical time periods through self-attention mechanisms, improving focus on important actions.

#### Theoretical Foundation
**Attention Hypothesis**: In the time series $\{\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_T\}$, different timesteps have different contributions to the final prediction, and there exists a learnable importance weight distribution.

**Multi-head Self-Attention Mechanism**:
Define query, key, value matrices:
$$\mathbf{Q} = \mathbf{H}W_Q, \quad \mathbf{K} = \mathbf{H}W_K, \quad \mathbf{V} = \mathbf{H}W_V$$

where $\mathbf{H} = [\mathbf{h}_1; \mathbf{h}_2; ...; \mathbf{h}_T] \in \mathbb{R}^{T \times d}$

**Attention Weight Computation**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Multi-head Mechanism**:
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

where:
$$\text{head}_i = \text{Attention}(\mathbf{Q}W_Q^i, \mathbf{K}W_K^i, \mathbf{V}W_V^i)$$

#### Detailed Architecture
```
Input Processing:
├── Input: [batch_size, seq_len=20, input_size=70]
├── Linear Projection Layer: 70 → 128
└── Positional Encoding (optional)

Bidirectional LSTM Encoder:
├── LSTM(input_size=128, hidden_size=64, bidirectional=True)
├── Output: [batch_size, seq_len, 128] (64×2)
└── Dropout(0.3)

Multi-head Self-Attention:
├── Number of heads: h=8
├── Dimension per head: d_k = d_v = 128/8 = 16
├── Query/Key/Value Projection:
│   ├── W_Q ∈ ℝ^(128×128)
│   ├── W_K ∈ ℝ^(128×128)  
│   └── W_V ∈ ℝ^(128×128)
├── Scaled Dot-Product Attention:
│   └── α_ij = softmax(Q_i·K_j^T / √16)
└── Output Projection: W_O ∈ ℝ^(128×128)

Residual Connection and Normalization:
├── Residual Connection: output = input + attention_output
├── Layer Normalization
└── Feed-Forward Network: 128 → 256 → 128

Global Pooling and Classification:
├── Global Average Pooling: [batch_size, seq_len, 128] → [batch_size, 128]
├── Fully Connected Layer: 128 → num_classes
└── Softmax Activation
```

#### Mathematical Derivation

**Meaning of Attention Weights**:

Attention weight represents the degree of attention position i pays to position j:

$$
\alpha_{ij} = \frac{\exp(\text{score}(\mathbf{h}_i, \mathbf{h}_j))}{\sum_{k=1}^T \exp(\text{score}(\mathbf{h}_i, \mathbf{h}_k))}
$$

**Theoretical Basis for Scaling Factor**:

When d_k is large, dot product values can be very large, pushing the softmax function into saturated regions. The scaling factor ensures gradient stability:

$$
\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_i) = d_k
$$

After scaling:

$$
\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = 1
$$

**Theoretical Advantages of Multi-head Attention**:
Different attention heads can focus on different relationship patterns:
- Head 1: Short-term dependencies (adjacent timesteps)
- Head 2: Medium-term patterns (local peaks)
- Head 3: Long-term trends (global patterns)

**Time Complexity Analysis**:
- Self-attention: $O(T^2 \cdot d)$
- LSTM: $O(T \cdot d^2)$
- Total complexity: $O(T^2 \cdot d + T \cdot d^2)$

For $T=20, d=128$: self-attention dominates

#### Key Innovations
1. **Dynamic Attention Allocation**: Adaptively identifies important time periods
2. **Multi-dimensional Relationship Modeling**: Multi-head mechanism captures different types of temporal patterns
3. **Long-range Dependencies**: Overcomes LSTM's long-term dependency problem
4. **Parallel Computation**: Attention mechanism supports parallelization, improving training efficiency

### 4. Feature-Selective Net: Adaptive Feature Selection Mechanism

#### Design Motivation
Among 70-dimensional sensor features, not all features are equally important for every activity. This architecture dynamically identifies and emphasizes feature subsets most relevant to the current sample through a learnable feature selection gating mechanism.

#### Theoretical Foundation

**Feature Importance Hypothesis**: For different nursing activities, feature importance varies significantly. Define feature importance vector g ∈ [0,1]^d, where g_i represents the importance weight of the i-th feature.

**Mathematical Definition of Gating Mechanism**:

$$
\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{x} + \mathbf{b}_g)
$$

where σ is the Sigmoid function, ensuring gating weights are in the [0,1] range.

**Feature Selection Operation**:

$$
\mathbf{x}_{\text{selected}} = \mathbf{g} \odot \mathbf{x}
$$

where ⊙ denotes element-wise multiplication.

#### Detailed Architecture
```
Feature Selection Gating Module:
├── Input: [batch_size, seq_len=20, features=70]
├── Global Average Pooling: [batch_size, seq_len, 70] → [batch_size, 70]
├── Feature Importance Network:
│   ├── Fully Connected Layer 1: 70 → 35 (feature compression)
│   ├── ReLU activation
│   ├── Dropout(0.3)
│   ├── Fully Connected Layer 2: 35 → 70 (feature recovery)
│   └── Sigmoid activation → gating weights g ∈ [0,1]^70
└── Gating Operation: x_gated = g ⊙ x (element-wise multiplication)

Main Network:
├── Input: Gated features [batch_size, seq_len, 70]
├── 1D Conv Layer 1: (filters=64, kernel=3)
│   ├── BatchNorm1d + ReLU
│   └── Dropout(0.3)
├── 1D Conv Layer 2: (filters=128, kernel=3)
│   ├── BatchNorm1d + ReLU
│   └── Dropout(0.3)
├── Global Average Pooling: [batch_size, seq_len, 128] → [batch_size, 128]
├── Fully Connected Layer: 128 → num_classes
└── Softmax activation
```

#### Mathematical Derivation

**Property Analysis of Gating Function**:
Derivative of Sigmoid function:
$$\frac{\partial \sigma(x)}{\partial x} = \sigma(x)(1 - \sigma(x))$$

When $\sigma(x) \to 0$ or $\sigma(x) \to 1$, the gradient approaches 0, achieving a "hard" selection effect.

**Information-Theoretic Interpretation of Feature Selection**:

Define information entropy after selection:

$$
H(\mathbf{x}_{\text{selected}}) = -\sum_{i=1}^d p(x_i) \log p(x_i)
$$

where

$$
p(x_i) = \frac{g_i |x_i|}{\sum_{j=1}^d g_j |x_j|}
$$

The goal is to maximize information entropy of relevant features while minimizing contribution of irrelevant features.

**Regularization of Gating Weights**:

To prevent over-sparsification, introduce L1 regularization term:

$$
\mathcal{L}_{\text{reg}} = \lambda \sum_{i=1}^d |g_i|
$$

Total loss function:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{reg}}
$$

#### Theoretical Advantages of Gating Mechanism

**1. Adaptability**:

Different samples activate different feature subsets:

$$
\mathbf{g}^{(n)} = f_{\text{gate}}(\mathbf{x}^{(n)})
$$

**2. Interpretability**:

Gating weights directly reflect feature importance, facilitating analysis:

$$
\text{Importance}(f_i) = \mathbb{E}[g_i]
$$

**3. Computational Efficiency**:

Feature selection reduces subsequent computation:

$$
\text{Complexity}_{\text{reduced}} = \text{Complexity}_{\text{original}} \times \mathbb{E}[\|\mathbf{g}\|_1/d]
$$

#### Key Innovations
1. **Dynamic Feature Selection**: Adaptively adjusts feature weights based on input samples
2. **End-to-End Learning**: Gating weights jointly optimized with main classifier
3. **Enhanced Interpretability**: Provides intuitive interpretation of feature importance
4. **Computational Efficiency**: Reduces computational overhead of irrelevant features

### 5. HybridNet: Integrated Multi-modal Learning Architecture

#### Design Motivation
Single techniques often only solve specific problems, while nursing activity recognition faces multiple challenges: feature redundancy, temporal complexity, physical constraints, etc. HybridNet integrates three complementary techniques through modular design to achieve collaborative optimization.

#### Theoretical Foundation
**Ensemble Learning Theory**: Assume there exist three independent feature transformation functions:
- $f_{\text{fs}}: \mathbb{R}^d \rightarrow \mathbb{R}^d$ (feature selection)
- $f_{\text{ca}}: \mathbb{R}^d \rightarrow \mathbb{R}^{d'}$ (correlation-aware)  
- $f_{\text{ta}}: \mathbb{R}^{T \times d'} \rightarrow \mathbb{R}^{d''}$ (temporal attention)

**Ensemble Mapping**:

$$
\mathbf{h}_{\text{hybrid}} = f_{\text{ta}}(f_{\text{ca}}(f_{\text{fs}}(\mathbf{X})))
$$

#### Detailed Architecture
```
HybridNet Complete Architecture:

Stage 1: Adaptive Feature Selection
├── Input: [batch_size, seq_len=20, features=70]
├── Global Context Extraction: GlobalAvgPool1d
├── Feature Importance Network:
│   ├── FC1: 70 → 35, ReLU, Dropout(0.3)
│   ├── FC2: 35 → 70, Sigmoid
│   └── Output: Gating weights g ∈ [0,1]^70
└── Gating Operation: X_fs = g ⊙ X

Stage 2: Physics-Aware Correlation Learning
├── Input: X_fs [batch_size, seq_len, 70]
├── Feature Grouping:
│   ├── G₁: Quaternions [0:12]
│   ├── G₂: Quaternion Derivatives [12:24]  
│   ├── G₃: Velocity [24:48]
│   └── G₄: Magnetic Field [48:70]
├── Grouped Convolution Processing:
│   ├── Each group: Conv1d(filters=32, kernel=3) + BatchNorm + ReLU
│   └── Output: 4 group features h⁽¹⁾, h⁽²⁾, h⁽³⁾, h⁽⁴⁾
├── Correlation Computation:
│   ├── Inter-group correlations: c_ij = corr(h⁽ⁱ⁾, h⁽ʲ⁾) for i≠j
│   └── Total 6 correlation features
└── Feature Fusion: X_ca = Concat[h⁽¹⁾, h⁽²⁾, h⁽³⁾, h⁽⁴⁾, c₁₂, c₁₃, c₁₄, c₂₃, c₂₄, c₃₄]

Stage 3: Temporal Attention Modeling
├── Input: X_ca [batch_size, seq_len, 320] (4×32 + 6×32)
├── Bidirectional LSTM Encoding:
│   ├── LSTM(input_size=320, hidden_size=64, bidirectional=True)
│   └── Output: [batch_size, seq_len, 128]
├── Multi-head Self-Attention:
│   ├── Number of heads: 8, dimension per head: 16
│   ├── Q,K,V projection: 128 → 128
│   ├── Scaled dot-product attention: softmax(QK^T/√16)V
│   └── Output projection: 128 → 128
├── Residual Connection: output = LSTM_out + Attention_out
├── Layer Normalization
└── Feed-Forward Network: 128 → 256 → 128

Final Classification:
├── Global Average Pooling: [batch_size, seq_len, 128] → [batch_size, 128]
├── Classifier:
│   ├── FC1: 128 → 64, ReLU, Dropout(0.5)
│   ├── FC2: 64 → num_classes
│   └── Softmax activation
└── Output: Class probability distribution
```

#### Mathematical Modeling

**Mathematical Representation of Modular Integration**:
Define configurable module selector:
$$\mathcal{M} = \{\alpha_{\text{fs}}, \alpha_{\text{ca}}, \alpha_{\text{ta}}\} \in \{0,1\}^3$$

**Conditional Execution**:

$$
\mathbf{X}_1 = \begin{cases}
f_{\text{fs}}(\mathbf{X}) & \text{if } \alpha_{\text{fs}} = 1 \\
\mathbf{X} & \text{otherwise}
\end{cases}
$$

$$
\mathbf{X}_2 = \begin{cases}
f_{\text{ca}}(\mathbf{X}_1) & \text{if } \alpha_{\text{ca}} = 1 \\
\mathbf{X}_1 & \text{otherwise}
\end{cases}
$$

$$
\mathbf{X}_3 = \begin{cases}
f_{\text{ta}}(\mathbf{X}_2) & \text{if } \alpha_{\text{ta}} = 1 \\
\mathbf{X}_2 & \text{otherwise}
\end{cases}
$$

**Joint Loss Function**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{fs}} + \lambda_2 \mathcal{L}_{\text{ca}} + \lambda_3 \mathcal{L}_{\text{ta}}
$$

where:
- L_CE: Cross-entropy loss
- L_fs = ||g||₁: Feature selection sparsity loss
- L_ca = Σ ||c_ij||₂²: Correlation regularization
- L_ta = ||A||_F²: Attention weight regularization

#### Inter-module Interaction Analysis

**1. Feature Selection → Correlation Awareness**:

Feature selection reduces noise, improving accuracy of correlation computation:

$$
\text{SNR}_{\text{improved}} = \frac{\text{Signal}_{\text{selected}}}{\text{Noise}_{\text{filtered}}}
$$

**2. Correlation Awareness → Temporal Attention**:

Structured features provide better foundation for temporal modeling:

$$
\text{Attention}_{\text{quality}} \propto \text{Feature}_{\text{structure}}
$$

**3. End-to-End Optimization**:

Gradients backpropagate through all modules:

$$
\frac{\partial \mathcal{L}}{\partial \theta_{\text{fs}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{X}_3} \cdot \frac{\partial \mathbf{X}_3}{\partial \mathbf{X}_2} \cdot \frac{\partial \mathbf{X}_2}{\partial \mathbf{X}_1} \cdot \frac{\partial \mathbf{X}_1}{\partial \theta_{\text{fs}}}
$$

#### Theoretical Advantages Analysis

**1. Complementarity**:
- Feature selection: Solves feature redundancy problem
- Correlation awareness: Utilizes physical structure information
- Temporal attention: Captures important temporal patterns

**2. Robustness**:

Modular design provides fault tolerance:

$$
P(\text{System Failure}) = \prod_{i=1}^3 P(\text{Module}_i \text{ Failure})
$$

**3. Extensibility**:

New modules can be seamlessly integrated:

$$
f_{\text{new}} = f_{\text{module}_n} \circ f_{\text{module}_{n-1}} \circ ... \circ f_{\text{module}_1}
$$

#### Key Innovations
1. **Unified Integration Framework**: Organic combination of three complementary techniques
2. **Modular Design**: Supports dynamic configuration and ablation studies  
3. **End-to-End Optimization**: All components jointly trained, avoiding suboptimal solutions
4. **Physics-Constraint Awareness**: Integrating sensor physical knowledge into deep learning
5. **Multi-level Feature Learning**: Hierarchical modeling from feature level to temporal level

## Experimental Methodology and Theoretical Analysis

### Data Leakage Prevention Strategy
**Theoretical Foundation**: Traditional random splitting causes same subject's data to be distributed across training, validation, and test sets, leading to temporal dependency leakage.

**Subject-level Splitting**:
$$\mathcal{S} = \{S_1, S_2, ..., S_{13}\} \rightarrow \{\mathcal{S}_{\text{train}}, \mathcal{S}_{\text{val}}, \mathcal{S}_{\text{test}}\}$$

**Mathematical Verification**:
Let $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$ at subject level, then:
$$P(\text{data leakage}) = P(\exists i,j : \text{subject}(x_i^{\text{train}}) = \text{subject}(x_j^{\text{test}})) = 0$$

### Temporal Dependency Elimination
**Problem Formalization**:
Correlation between adjacent windows in time series:
$$\rho(W_i, W_{i+1}) = \frac{\text{Cov}(W_i, W_{i+1})}{\sigma(W_i)\sigma(W_{i+1})}$$

**Solution**:
1. **Non-overlapping Windows**: Step size $s = 2W$, ensuring $W_i \cap W_j = \emptyset$ for $|i-j| \geq 1$
2. **Temporal Order Shuffling**: Random permutation of training windows breaks temporal patterns
3. **Intra-window Consistency**: Only retain windows with completely identical labels

### Overfitting Prevention Strategy
**Theoretical Basis**: Deep networks easily overfit on high-dimensional sparse data, requiring multi-level regularization.

```
Regularization Technique Combination:
├── Data Level:
│   ├── Label Smoothing: y_soft = (1-ε)y_hard + ε/K
│   ├── Input Perturbation: x_aug = x + N(0, σ²)
│   └── Temporal Window Augmentation: Random start point sampling
├── Model Level:
│   ├── Dropout: p(x_i = 0) = p_drop
│   ├── BatchNorm: x_norm = (x-μ)/σ
│   └── Weight Decay: L2 regularization λ||θ||²
├── Optimization Level:
│   ├── Gradient Clipping: ||∇θ|| ≤ τ
│   ├── Learning Rate Scheduling: lr × γ when plateau
│   └── Early Stopping: monitor val_loss patience
└── Loss Level:
    ├── Label Smoothing: Reduce overconfidence
    ├── Focal Loss: Focus on hard samples
    └── Multi-task Learning: Feature-level auxiliary loss
```

**Mathematical Modeling**:
Total regularization loss:
$$\mathcal{L}_{\text{reg}} = \lambda_1\|\theta\|_2^2 + \lambda_2\|\theta\|_1 + \lambda_3 H(\text{predictions})$$

where $H(\cdot)$ is prediction entropy, encouraging moderate uncertainty.

### Training Configuration

### Data Processing Pipeline
```
Preprocessing Pipeline:
├── Subject-level data splitting (prevent data leakage)
├── Class balance check (min_samples=5000)
├── Feature standardization (based on training set)
├── Temporal window creation (window_size=20, non-overlapping)
├── Temporal order shuffling (break temporal dependencies)
└── Batch loading (batch_size=8)
```

## Ablation Study

### Research Purpose
Systematically evaluate the contribution of each component in HybridNet to model performance, understanding the mechanism of different techniques.

### Ablation Configurations

**Testing Configuration Matrix:**

| Configuration Name | Feature Selection | Correlation Aware | Temporal Attention |
|-------------------|:-----------------:|:-----------------:|:------------------:|
| Baseline (No Components) | ❌ | ❌ | ❌ |
| Feature Selection Only | ✅ | ❌ | ❌ |
| Correlation Aware Only | ❌ | ✅ | ❌ |
| Temporal Attention Only | ❌ | ❌ | ✅ |
| Feature + Correlation | ✅ | ✅ | ❌ |
| Feature + Attention | ✅ | ❌ | ✅ |
| Correlation + Attention | ❌ | ✅ | ✅ |
| Full HybridNet | ✅ | ✅ | ✅ |

### Evaluation Metrics
- **Performance Metrics**: Test accuracy, F1-score
- **Efficiency Metrics**: Training time, convergence epochs
- **Contribution Analysis**: Performance improvement relative to baseline

### Visualization Outputs
- Performance ranking charts
- Component contribution heatmaps
- Complexity vs. performance scatter plots
- Training efficiency comparison

## Usage

### Environment Requirements
```bash
pip install torch torchvision
pip install scikit-learn pandas numpy
pip install matplotlib seaborn
```

### Quick Start
```bash
# 1. Run complete experiment
cd code
python run.py

# 2. View logs
tail -f ../logs/experiment_log_*.txt

# 3. View results
ls ../results/
```

### Configuration Options
```python
CONFIG = {
    'min_samples_per_class': 5000,  # Minimum samples per activity class
    'max_files': 253,               # Maximum number of files
    'include_ablation': True,       # Whether to include ablation study
}
```

## Computational Complexity Analysis

### Model Complexity Comparison
**Parameter Count Analysis**:
```
Model Parameter Statistics:
├── Baseline CNN-LSTM:      ~2.1M parameters
├── Correlation-Aware CNN:  ~1.8M parameters (reduced by grouped convolution)
├── Attention LSTM:         ~3.2M parameters (increased by attention mechanism)
├── Feature-Selective Net:  ~2.3M parameters (gating network overhead)
└── HybridNet:             ~4.1M parameters (integrated architecture)
```

**Time Complexity**:
Let input dimension be $d=70$, sequence length $T=20$, hidden dimension $h=128$

| Model | Training Complexity | Inference Complexity | Main Bottleneck |
|------|-----------|-----------|----------|
| Baseline | $O(Th^2 + Td^2)$ | $O(Th^2)$ | LSTM computation |
| Correlation-Aware | $O(Td^2/G)$ | $O(Td^2/G)$ | Grouped convolution |
| Attention | $O(T^2h + Th^2)$ | $O(T^2h)$ | Attention matrix |
| Feature-Selective | $O(Td^2 + d^2)$ | $O(Td^2)$ | Gating computation |
| HybridNet | $O(T^2h + Td^2)$ | $O(T^2h)$ | Overall complexity |

**Space Complexity**:
- Feature storage: $O(BTd)$ where $B$ = batch size
- Intermediate activations: Different memory usage patterns for each model
- Gradient storage: Linearly related to parameter count

## Output Results

### Automatically Generated Files
```
../results/
├── pytorch_experimental_results.csv      # Model performance comparison table
├── detailed_experimental_results.json    # Detailed results JSON
├── comprehensive_results.png             # Comprehensive performance visualization
├── training_curves.png                   # Training curve plots
├── all_confusion_matrices.png            # All confusion matrices
├── training_histories/                   # Training histories
│   ├── Baseline_CNN-LSTM_history.json
│   ├── Attention_LSTM_history.json
│   └── ...
├── confusion_matrices/                   # Individual confusion matrices
├── roc_curves/                          # ROC curve plots
└── ablation_study/                      # Ablation study results
    ├── ablation_results.json
    ├── ablation_summary.csv
    └── ablation_visualization.png
```

### Performance Evaluation and Interpretation
```
Evaluation Metrics:
├── Accuracy
├── F1-Score 
├── Precision
├── Recall
├── Training Time
├── Overfitting Analysis (Train-Val Gap)
└── ROC-AUC (multi-class)
```

## Experimental Results Interpretation

### Performance Metric Guidelines
- **Train-Val Gap < 0.2**: Model generalizes well
- **Train-Val Gap > 0.2**: Overfitting risk exists
- **Val Accuracy > 80%**: Excellent performance
- **Val Accuracy 60-80%**: Good performance
- **Val Accuracy < 60%**: Needs improvement

### Common Issues Troubleshooting
1. **Overfitting**: Training accuracy much higher than validation accuracy
   - Solution: Increase regularization, reduce model complexity
2. **Underfitting**: Both training and validation accuracy are low
   - Solution: Increase model capacity, adjust learning rate
3. **Class Imbalance**: Some classes have very low recognition rates
   - Solution: Class weight balancing, data augmentation

## Technical Features

### Innovations
1. **Multi-technique Fusion**: First to combine feature selection, correlation awareness, and temporal attention
2. **Overfitting Protection**: Comprehensive regularization strategy ensures model generalization
3. **Systematic Ablation**: Detailed analysis of each component's contribution
4. **Strong Practicality**: Validated on real nursing data, deployable application

### Application Value
- **Intelligent Nursing**: Automatically recognize nursing activities, improve nursing quality
- **Health Monitoring**: Real-time activity monitoring, timely detection of anomalies
- **Research Tool**: Provide objective data support for nursing research

---

## License

MIT License - See LICENSE file for details

## Contact

For questions or suggestions, please contact: [xiaoyanli629@tsinghua.edu.cn]

---

*This comprehensive documentation serves as a complete reference for the deep learning models and methodologies developed for nursing activity recognition, providing both theoretical foundations and practical implementation details for researchers and practitioners in the field.*

**Last Updated**: Nov 4, 2025
