# Building a Neural Network From Scratch — Documentation

A record of everything built in this project, why each piece exists, and how they connect. Written for a CPU-and-RAM-only setup (no GPU) — every design choice below accounts for that constraint.

---

## 1. The core idea, in one sentence

A neural network is a math function that starts knowing nothing (random numbers) and gradually corrects itself by comparing its guesses to the right answers, over and over, using two mechanisms: **backpropagation** (figuring out which direction to correct each number) and **gradient descent** (actually making the correction).

Everything built in this project is one continuous elaboration of that single idea.

---

## 2. The journey, stage by stage

### Stage 1 — Single neuron (`y = w·x + b`)

**What it is**: The smallest possible unit. Takes one input, multiplies it by a weight `w`, adds a bias `b`.

**Why it matters**: This is the atomic building block of every neural network in existence, including GPT-scale models. A "neuron" is never more complicated than this at its core — bigger networks just have more of them, wired together.

**Key mechanism learned**: Gradient descent — repeatedly nudging `w` and `b` in the direction that reduces error, like a ball rolling downhill on a loss surface.

**Important discovery**: Multiple parameters don't move independently toward their "true" values — they co-adapt, sometimes overshooting or moving temporarily in the "wrong" direction before correcting, because they're jointly minimizing one shared error signal.

### Stage 2 — Two neurons feeding an output neuron

**What it is**: Two hidden neurons process the same input in parallel, and a third "output" neuron combines their results.

**Why it matters**: This is the first real *layer* — several neurons doing the same kind of computation side by side. It's also where the chain rule gets genuinely deeper: gradients for the hidden neurons are now *two steps* removed from the error, requiring the actual chain rule (multiplying local derivatives together) rather than one direct calculation.

**Important discovery**: Stacking multiple *linear* neurons doesn't add any real learning power — a linear function of a linear function is still just linear. This directly motivated the next stage.

### Stage 3 — ReLU activation function

**What it is**: A small non-linear twist applied after each neuron: output the input if positive, otherwise output `0`.

**Why it matters**: This is what actually allows a network to bend and fit curves, rather than being permanently limited to straight lines. Every meaningful capability of deep learning — recognizing images, understanding language, anything non-trivial — depends on stacking non-linear activations. Without them, no amount of extra neurons or layers adds any real power.

**Important discovery (the hard way)**: "Dying ReLU" — if a neuron's inputs push it permanently negative, its gradient becomes exactly `0` forever, and it can never recover. This is a real, well-documented failure mode in production deep learning, not just a toy problem. It was fixed here with: more neurons (safety in numbers), smaller learning rate, and better weight initialization.

### Stage 4 — Autograd engine (`Value.ts`)

**What it is**: Instead of hand-deriving every gradient formula (like `2 * error * x`), this is a small system that automatically tracks every mathematical operation as it happens, then walks backward through that record applying the chain rule — computing every gradient in the entire network with one function call: `.backward()`.

**Why it matters**: This is *exactly* the mechanism that sits underneath PyTorch, TensorFlow, and every modern deep learning framework. There is no additional secret sauce — production libraries add more supported operations, GPU execution, and heavy engineering, but the core idea (build a graph, walk it backward, accumulate gradients) is identical to what's in `Value.ts`. This stage was specifically built to replace "trust the library" with "understand exactly what the library does."

**How it was verified**: Every gradient the engine computed was checked against gradients hand-derived with calculus, and they matched exactly.

### Stage 5 — Tensor: batching with arrays and matrix multiplication (`Tensor.ts`)

**What it is**: The same autograd idea as `Value.ts`, but every operation now works on whole arrays/matrices of numbers at once, using **matrix multiplication (matmul)** as the core operation instead of a `for` loop over individual samples.

**Why it matters**: This is the single biggest lever for performance without a GPU. A `for` loop processes one number at a time; a matrix multiply processes an entire batch in one structured operation that a computer's math libraries can execute far more efficiently — this is genuinely how NumPy, PyTorch, and TensorFlow get their speed, even on CPU.

**How it was verified**: Gradient-checked against numerical (finite-difference) derivatives — a gold-standard technique for confirming an autograd/tensor implementation is mathematically correct, not just "seems to work."

### Stage 6 — CPU/RAM-specific optimizations

Since there's no GPU available, every choice from this point on was made specifically for CPU + RAM efficiency:

- **`Float32Array` instead of plain arrays**: JavaScript numbers are 64-bit by default. Using 32-bit floats halves memory use and keeps numbers packed contiguously in memory, which CPUs read much faster than scattered values.
- **Batched (matrix) operations instead of loops**: covered in Stage 5 — this is the difference between "technically correct but slow" and "actually efficient" on a CPU.
- **Fixed-size, small architectures**: deliberately kept the network small (a handful of neurons) rather than attempting anything at real-world scale — appropriate for learning, and realistic for CPU-only hardware.

---

## 3. Why GPUs matter (and why this project didn't need one — yet)

| | CPU | GPU |
|---|---|---|
| Design goal | A few very fast, flexible cores | Thousands of simple cores |
| Best at | Sequential, varied tasks | The *same* operation repeated across huge amounts of data |
| Neural net fit | Fine for small models (like everything built here) | Ideal — every neuron in a layer does the same multiply-add, just with different numbers, which GPUs can do in parallel |

Training something like GPT is the exact same forward pass → loss → backward pass → gradient descent loop built here — just with billions of parameters, which makes GPU-style parallelism a necessity rather than a nice-to-have. For learning the fundamentals and for small models, CPU + RAM is genuinely sufficient — which is exactly what this project has proven, end to end.

---

## 4. Practical playbook for CPU/RAM-only development going forward

1. **Prefer batched matrix operations over loops** wherever possible (Stage 5's biggest lesson).
2. **Use `Float32Array`/typed arrays**, not plain JS number arrays, for anything performance-sensitive.
3. **Keep models and datasets appropriately small** — this is not a limitation to apologize for, it's the correct scope for learning and for many real small-scale problems.
4. **When ready to scale up**: reach for a real library (TensorFlow.js, or Python + NumPy/PyTorch) that has proper CPU-level vectorization (SIMD) built in — the concepts transfer directly, since it's the same `Value`/`Tensor` idea, just professionally engineered.
5. **When a project genuinely outgrows CPU capacity**: free cloud GPU options (like Google Colab) exist specifically for this transition point, with no local hardware needed.

---

## 5. What's actually been built, technically

- `Value.ts` — scalar autograd engine (add, mul, sub, pow, relu, backward via topological sort)
- `Tensor.ts` — array/matrix autograd engine (matmul, add with broadcasting, relu, sumSquares, backward)
- `MainTensor.ts` — a fully batched, ReLU-activated, 2-layer neural network trained with matrix operations and automatic differentiation, successfully fitting a non-linear curve (`y = x²`) from randomly initialized weights

This is, in miniature, structurally identical to what sits underneath every production deep learning framework — the difference is scale and engineering polish, not the underlying idea.
