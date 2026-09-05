# nanograd

A tiny neural network and autograd engine, built completely from scratch in TypeScript — no ML libraries, no black boxes, CPU-only. Every gradient, every weight update, every matrix multiply is hand-written and verified against numerical differentiation.

This isn't a wrapper around TensorFlow or PyTorch. It's the same core mechanism those libraries use internally — a computation graph with automatic backpropagation — built small enough to read top to bottom in one sitting.

## Why this exists

Most people learn deep learning by calling `model.fit()`. This project goes the other direction: building the actual mechanics (gradient descent, backpropagation, activation functions, automatic differentiation, batched tensor operations) one layer of abstraction at a time, so nothing is trusted without being understood.

## What's inside

| File | What it is |
|---|---|
| `Value.ts` | A minimal scalar autograd engine — tracks operations on single numbers and computes gradients automatically via backpropagation |
| `Tensor.ts` | The same autograd idea extended to arrays/matrices, with batched `matmul`, `add`, `relu` — this is what makes training more than one sample at a time efficient |
| `MainTensor.ts` | A working 2-layer neural network (ReLU hidden layer + linear output) trained end-to-end on a non-linear curve (`y = x²`), using `Tensor.ts` |
| `notes.md` | Full write-up of the concepts, in order: single neuron → multiple neurons → activation functions → autograd → batched tensors → CPU-specific optimizations |

## Quick start

Requires Node.js 22.6+ (native TypeScript support, no build step or extra tooling needed):

```bash
node MainTensor.ts
```

You should see the loss shrink over 5000 epochs, followed by predictions on unseen inputs the model wasn't trained on.

> If you're on an older Node version, install [tsx](https://github.com/privatenumber/tsx) instead: `npm install -g tsx && tsx MainTensor.ts`

## The core idea in one snippet

Every operation in the autograd engine remembers how it was computed, so gradients never need to be derived by hand:

```typescript
const a = new Value(6);
const b = new Value(2);
const f = a.mul(b);
f.backward();

console.log(a.grad); // automatically computed: 2
console.log(b.grad); // automatically computed: 6
```

Scale that same idea up to matrices, stack a couple of layers with a ReLU in between, and you have a genuine (if small) neural network — no library required.

## What this project deliberately does *not* have

- GPU acceleration (built and verified entirely on CPU + RAM)
- Support for large datasets or production-scale models
- Every operation type a real framework supports — just enough (`add`, `mul`, `matmul`, `relu`, `pow`) to train a real, working network

## Background

Built as a learning exercise to understand backpropagation and automatic differentiation from first principles, rather than treating deep learning frameworks as black boxes. See `neural-network-from-scratch.md` for the full walkthrough, including the mistakes made along the way (dying ReLU, gradient mismatches, co-adapting parameters) and what they revealed.

## License

MIT — do whatever you'd like with this.