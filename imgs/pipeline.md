```mermaid
%%{init: {'look': 'handDrawn', 'theme': 'default'}}%%
flowchart LR
    %% --- Styling ---
    classDef rep shape:hexagon,fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#1b4332
    classDef action shape:rounded,fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,color:#8a3b12

    %% --- Nodes ---
    Source{{High-level ML framework}}:::rep
    Input{{ONNX model}}:::rep
    Output{{ST program}}:::rep

    Shape("Shape inference"):::action
    BuildIR("Layer extraction and typing"):::action
    Order("Execution ordering\n(topological)"):::action
    Regionize("SCC-based regionization"):::action
    Optimize("Region optimization"):::action
    Lower("Region lowering and ST assembly"):::action

    %% --- Connections ---
    Source -->|Export| Input
    Input --> Shape

    subgraph Toolchain ["Compiler Toolchain"]
        direction LR
        subgraph IRBuildGroup ["IR construction"]
            direction LR
            Shape --> BuildIR --> Order
        end

        subgraph RegionFlowGroup ["Region-aware compilation"]
            direction LR
            Order --> Regionize --> Optimize --> Lower
        end
    end

    style Toolchain fill:#eaf2ff,stroke:#2a5db0,stroke-width:1.5px
    style IRBuildGroup fill:#f4f8ff,stroke:#2a5db0,stroke-width:1px
    style RegionFlowGroup fill:#eaf2ff,stroke:#2a5db0,stroke-width:1px

    Lower --> Output
```
