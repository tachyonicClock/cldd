## Applications

- Weight Regularization/Functional Regularization:
    - @drift replace anchor model with the snapshot
        - Question: compare exponential moving average against concept anchored.
- Replay:
    - @drift store examples in a new buffer
        - **Question: Compare reservoir sampling against 'concept balanced' sampling in
          highly imbalanced datasets.**
- Architecture Based:
    - Parameter Isolation:
        - @warn prune and tune
        - @drift freeze and train
    - Dynamic
        - @drift add new parameters
            - online lora


## Potential Streams to Detect Concept Drift on

- With feedback from error on stream (this is what my prototype is)
- With feedback from error on validation buffer contents.
- **Unsupervised. Change detection in embedding space.**
- Class labels. The class indices change overtime.
    - Not so interesting. Maybe I should limit my exploration to domain incremental
      invalidating this approach as viable.