# Research Methodology Diagram (16:9 Format)

```mermaid
flowchart LR
    Start([Research<br/>Methodology]) --> Dataset

    subgraph Dataset[1.Dataset Construction]
        direction TB
        A1[Human News<br/>Reuters, AP, BBC] --> A2[Synthetic<br/>GPT-4o]
    end

    Dataset --> Features

    subgraph Features[2 & 3.Feature Engineering]
        direction TB

        subgraph Stylo[Stylometric]
            B1[Lexical Diversity]
            B2[Sentence Complexity]
            B3[N-gram Patterns]
            B4[Punctuation]
        end

        subgraph Semantic[Semantic Fluctuation ★]
            C1[Topic Coherence]
            C2[Entity Consistency]
            C3[Temporal Logic]
            C4[Source Attribution]
            C5[Factual Grounding]
        end
    end

    Features --> Model

    subgraph Model[4.Model Development]
        direction TB
        D1[Ensemble Classifier]
    end

    Model --> Eval

    subgraph Eval[5.Evaluation]
        direction TB
        E1[Benchmark<br/>GPTZero]
        E2[Out-of-Distribution<br/>New LLMs]
    end

    Eval --> End([Results])

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style Semantic fill:#fff4e1,stroke:#ff9800,stroke-width:3px
    style Features fill:#f5f5f5

    classDef novelBox fill:#fff4e1,stroke:#ff9800,stroke-width:2px
```

## Visualization Instructions

To view this diagram:

1. **GitHub/GitLab**: View this markdown file directly - Mermaid is natively supported
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Copy the mermaid code to https://mermaid.live/
4. **Export**: Use mermaid-cli to export as PNG/SVG:
   ```bash
   mmdc -i research_methodology.md -o methodology_diagram.png -w 1920 -H 1080
   ```

## Diagram Key

- **Blue boxes**: Start/End points
- **Orange highlighted box with ★**: Novel contribution (Semantic Fluctuation Analysis)
- **Horizontal flow**: Optimized for 16:9 presentation format
- **5 main stages**: Dataset → Features → Model → Evaluation → Results
