# TabularSSLModel Logical Flow Diagram

## 🔄 Complete Model Flow

```mermaid
flowchart TD
    %% Initialization Flow
    A[TabularSSLModel.__init__] --> B{corruption provided?}
    B -->|Yes| C[is_ssl = True]
    B -->|No| D[is_ssl = False]
    C --> E[Store components: encoder, corruption, custom_loss_fn]
    D --> E
    E --> F[No head pre-initialization<br/>✅ Simplified!]
    
    %% Training Entry Point
    G[training_step] --> H{is_ssl?}
    H -->|True| I[_ssl_training_step]
    H -->|False| J[_standard_training_step]
    
    %% Standard Training Flow
    J --> K[Extract features & targets]
    K --> L[forward pass]
    L --> M[Cross-entropy loss]
    M --> N[Log & return loss]
    
    %% SSL Training Flow
    I --> O[Extract data from batch]
    O --> P[Apply corruption]
    P --> Q[corruption.forward]
    Q --> R[Get corrupted_data dict]
    R --> S[Encode corrupted data]
    S --> T[encode pipeline]
    T --> U{custom_loss_fn provided?}
    
    %% Custom Loss Path
    U -->|Yes| V[_compute_custom_loss]
    V --> W[Try full signature call]
    W --> X{TypeError?}
    X -->|No| Y[Return loss]
    X -->|Yes| Z[_create_predictions]
    Z --> AA[Create simple_head if needed]
    AA --> BB[Handle tensor dimensions]
    BB --> CC[Call with predictions, targets]
    CC --> Y
    
    %% Built-in Loss Path  
    U -->|No| DD[_compute_builtin_loss]
    DD --> EE[Check corruption class name]
    EE --> FF{Match found?}
    FF -->|vime| GG[vime_loss_fn]
    FF -->|scarf| HH[scarf_loss_fn]
    FF -->|recontab| II[recontab_loss_fn]
    FF -->|No match| JJ[_generic_reconstruction_loss]
    
    GG --> KK[Dynamic head creation<br/>in loss function]
    HH --> KK
    II --> KK
    JJ --> LL[Create reconstruction_head<br/>if needed]
    
    KK --> Y
    LL --> MM[MSE reconstruction loss]
    MM --> Y
    
    %% Encoding Pipeline
    T --> NN[Optional: embedding_layer]
    NN --> OO[event_encoder - Required]
    OO --> PP[Optional: sequence_encoder]
    PP --> QQ[Optional: projection_head]
    QQ --> RR[Return representations]
    RR --> U
    
    %% Final Steps
    Y --> SS[Log train/ssl_loss]
    SS --> TT[Return loss]
    N --> TT
    
    %% Styling
    classDef initClass fill:#e1f5fe
    classDef sslClass fill:#f3e5f5
    classDef standardClass fill:#e8f5e8
    classDef lossClass fill:#fff3e0
    classDef encodeClass fill:#fce4ec
    
    class A,B,C,D,E,F initClass
    class I,O,P,Q,R,S,U,SS,TT sslClass
    class J,K,L,M,N standardClass
    class V,W,X,Y,Z,AA,BB,CC,DD,EE,FF,GG,HH,II,JJ,KK,LL,MM lossClass
    class T,NN,OO,PP,QQ,RR encodeClass
```

## 🎯 Key Simplifications Highlighted

### ✅ Removed Complexity
- **No head pre-initialization** in constructor
- **Dynamic head creation** only when needed in loss functions
- **Unified interface** for any loss function
- **Clean auto-detection** with dictionary mapping

### 🔄 Streamlined Flows

#### 1. **Initialization Flow (Top)**
- Simple component storage
- SSL mode detection based on corruption presence
- No complex head initialization logic

#### 2. **Training Dispatch (Middle Left)**
- Clean branch between SSL and standard training
- Single entry point with clear routing

#### 3. **SSL Training Flow (Center)**
- Linear flow: data → corruption → encoding → loss
- Clear separation of custom vs built-in loss paths

#### 4. **Loss Computation (Right Side)**
- **Custom path**: Try full signature, fallback to simple
- **Built-in path**: Auto-detect corruption type, use appropriate loss
- **Fallback**: Generic reconstruction for unknown types

#### 5. **Encoding Pipeline (Bottom)**
- Clear sequence of optional and required components
- Consistent interface throughout

## 📊 Flow Characteristics

### 🟢 **Green (Standard Training)**: Simple supervised learning path
### 🟣 **Purple (SSL Training)**: Self-supervised learning orchestration  
### 🟠 **Orange (Loss Computation)**: Flexible loss function handling
### 🟥 **Pink (Encoding)**: Representation learning pipeline
### 🔵 **Blue (Initialization)**: Setup and configuration

## 🚀 Benefits of This Design

1. **Clear Separation**: Each flow has distinct responsibilities
2. **Unified Interface**: Single entry point handles any loss function
3. **Dynamic Adaptation**: Components created when needed with correct dimensions
4. **Extensible**: Easy to add new corruption types or loss functions
5. **Maintainable**: Clear boundaries make debugging and testing easier

The diagram shows how the simplified TabularSSLModel achieves **maximum flexibility with minimum complexity**! 🎉 