```mermaid
%%{init: {'look': 'handDrawn', 'theme': 'default'}}%%
flowchart LR

    %% --- Styling ---
    classDef rep shape:hexagon,fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#1b4332
    classDef action shape:rounded,fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,color:#8a3b12

    %% --- Nodes ---
    Input{{ONNX model}}:::rep
    Output{{ST program}}:::rep

    %% --- IR construction ---
    Canon("Normalization"):::action
    Infer("Shape + type inference"):::action
    Extract("Layer extraction"):::action
    Order("Dependency scheduling\n(SCC-aware)"):::action

    %% --- Optimization ---
    Regionize("Regionization\n(SCC-based)"):::action
    Optimize("Region optimization"):::action

    %% --- Codegen ---
    Lower("Region-aware lowering + ST assembly"):::action

    %% --- Connections ---
    Input --> Canon

    subgraph Toolchain ["Compiler Toolchain"]
        direction LR

        subgraph IRStage ["IR Construction"]
            Canon --> Infer --> Extract --> Order
        end

        subgraph OptStage ["Optimization"]
            Order --> Regionize --> Optimize
        end

        subgraph CodegenStage ["Code Generation"] 
            Optimize --> Lower
        end
    end

    style Toolchain fill:#eaf2ff,stroke:#2a5db0,stroke-width:1.5px
    style IRStage fill:#f4f8ff,stroke:#2a5db0,stroke-width:1px
    style OptStage fill:#eaf2ff,stroke:#2a5db0,stroke-width:1px
    style CodegenStage fill:#f4f8ff,stroke:#2a5db0,stroke-width:1px

    Lower --> Output
```
