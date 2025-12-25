# STM32 Touch Gesture Neural Network (Embedded Inference)

This project implements a **lightweight neural network inference pipeline on STM32** for classifying touch/gesture inputs from three sensors.  
The neural network is **trained offline in Python/TensorFlow** and deployed to the STM32 as **plain C code** for fast, deterministic execution.

This README documents **only the core embedded logic** and runtime behavior.

---

## Overview

- Platform: **STM32 (bare metal / HAL compatible)**
- Inputs: **3 normalized sensor values** (`s1`, `s2`, `s3`)
- Output: **Gesture class ID**
- Inference: **Fully connected neural network in C**
- No dynamic memory allocation
- No floating-point dependencies beyond `float`

---

## Neural Network Summary

- **Architecture:**  
  ```
  Input (3) → Dense (5, ReLU) → Dense (8, Linear)
  ```

- **Activation Functions**
  - Hidden layer: ReLU
  - Output layer: Linear (logits)

- **Output Interpretation**
  - The output layer produces **logits**
  - The predicted class is the **index of the maximum logit**

This matches the TensorFlow model trained with:
```
SparseCategoricalCrossentropy(from_logits=True)
```

---

## Gesture Classes

The network outputs one of **8 gesture classes**, indexed in a fixed order that must match training:

| Class ID | Gesture Name      |
|--------:|-------------------|
| 0 | Light Touch |
| 1 | Hard Touch |
| 2 | Left Light |
| 3 | Left Hard |
| 4 | Right Light |
| 5 | Right Hard |
| 6 | Middle |
| 7 | Indeterminate |

> ⚠️ **Important:**  
> Class ordering **must not change** unless the network is retrained.

---

## Input Requirements

- Sensor values **must be normalized to `0.0 – 1.0`**
- Inputs are passed as:
  ```c
  float input[3] = { s1, s2, s3 };
  ```

Examples:
```
s1=0.24, s2=0.02, s3=0.01   → Left Light
s1=0.82, s2=0.81, s3=0.79   → Hard Touch
```

---

## Inference Flow

1. STM32 acquires sensor readings
2. Values are normalized to `0..1`
3. Input vector is passed to the NN
4. Hidden layer computation (matrix multiply + bias + ReLU)
5. Output layer computation (matrix multiply + bias)
6. Argmax over output logits
7. Gesture ID returned

All computations are **fully deterministic**.

---

## C API Overview

### Neural Network Interface

Declared in `nn.h`:

```c
int nn_predict(const float input[3]);
```

- **Input:** `float[3]` normalized sensor values
- **Return:** Integer gesture class ID (`0–7`)

---

## Integration Example

```c
float input[3] = { s1, s2, s3 };
int gesture = nn_predict(input);

switch (gesture) {
    case 0: /* Light Touch */ break;
    case 1: /* Hard Touch */ break;
    case 2: /* Left Light */ break;
    case 3: /* Left Hard */ break;
    case 4: /* Right Light */ break;
    case 5: /* Right Hard */ break;
    case 6: /* Middle */ break;
    case 7: /* Indeterminate */ break;
}
```

---

## Design Goals

- ✔ Deterministic execution
- ✔ No heap usage
- ✔ No CMSIS-NN or DSP dependencies
- ✔ Easy weight regeneration from Python
- ✔ Debuggable with standard C tools

---

## Updating the Network

To update the model:

1. Retrain the network in Python
2. Export new weight arrays (`W1`, `B1`, `W2`, `B2`)
3. Replace weights in `nn.c`
4. Rebuild firmware

No other code changes are required.

---

## Notes

- This is **inference-only** code
- Training is intentionally excluded from the firmware
- The model is small enough to run at high frequency in main loop or ISR-safe context

---

## License

MIT (or project-specific)
