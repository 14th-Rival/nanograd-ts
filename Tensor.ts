// Tensor: the array-based version of Value. Same core idea (remember how you
// were computed, walk backward, apply the chain rule) but every operation now
// works on a whole grid of numbers at once instead of a single number.
//
// Data is stored FLAT (a single number[]) plus a `shape` describing how to
// read it as rows/columns. This flat-array-plus-shape layout is exactly how
// real libraries (NumPy, PyTorch, TensorFlow) store data internally -- it's
// what makes batch operations fast, because the numbers sit next to each
// other in memory instead of being scattered across nested arrays.

export class Tensor {
    public data: Float32Array;
    public grad: Float32Array;
    public shape: [number, number]; // [rows, cols] -- rows = batch size, cols = features
    private _backward: () => void = () => {};
    private _prev: Set<Tensor>;

    constructor(data: number[] | Float32Array, shape: [number, number], children: Tensor[] = []) {
        this.data = Float32Array.from(data);
        this.shape = shape;
        this.grad = new Float32Array(this.data.length); // starts at all zeros
        this._prev = new Set(children);
    }

    static zeros(shape: [number, number]): Tensor {
        return new Tensor(new Float32Array(shape[0] * shape[1]), shape);
    }

    static randomUniform(shape: [number, number], min = -1, max = 1): Tensor {
        const n = shape[0] * shape[1];
        const data = Array.from({ length: n }, () => Math.random() * (max - min) + min);
        return new Tensor(data, shape);
    }

    private idx(r: number, c: number): number {
        return r * this.shape[1] + c;
    }

    // --- MATRIX MULTIPLY ---
    // This is THE operation that replaces nested for-loops with one batched
    // computation. (rows x k) @ (k x cols) -> (rows x cols).
    // It is also the single most important operation for GPU acceleration --
    // GPUs are built almost specifically to make this one operation fast.
    matmul(other: Tensor): Tensor {
        const [r, k] = this.shape;
        const [k2, c] = other.shape;
        if (k !== k2) throw new Error(`Shape mismatch: (${r}x${k}) @ (${k2}x${c})`);

        const outData = new Float32Array(r * c);
        for (let i = 0; i < r; i++) {
            for (let j = 0; j < c; j++) {
                let sum = 0;
                for (let t = 0; t < k; t++) {
                    sum += this.data[this.idx(i, t)] * other.data[other.idx(t, j)];
                }
                outData[i * c + j] = sum;
            }
        }
        const out = new Tensor(outData, [r, c], [this, other]);

        out._backward = () => {
            // dL/dA = dL/dOut @ B^T   and   dL/dB = A^T @ dL/dOut
            for (let i = 0; i < r; i++) {
                for (let t = 0; t < k; t++) {
                    let g = 0;
                    for (let j = 0; j < c; j++) {
                        g += out.grad[i * c + j] * other.data[other.idx(t, j)];
                    }
                    this.grad[this.idx(i, t)] += g;
                }
            }
            for (let t = 0; t < k; t++) {
                for (let j = 0; j < c; j++) {
                    let g = 0;
                    for (let i = 0; i < r; i++) {
                        g += this.data[this.idx(i, t)] * out.grad[i * c + j];
                    }
                    other.grad[other.idx(t, j)] += g;
                }
            }
        };
        return out;
    }

    // --- ADD, with broadcasting for a bias row (shape [1, cols]) ---
    add(other: Tensor): Tensor {
        const broadcastRows = other.shape[0] === 1 && this.shape[0] !== 1;
        const [r, c] = this.shape;
        const outData = new Float32Array(r * c);
        for (let i = 0; i < r; i++) {
            for (let j = 0; j < c; j++) {
                const ov = broadcastRows ? other.data[j] : other.data[this.idx(i, j)];
                outData[this.idx(i, j)] = this.data[this.idx(i, j)] + ov;
            }
        }
        const out = new Tensor(outData, [r, c], [this, other]);
        out._backward = () => {
            for (let i = 0; i < r * c; i++) this.grad[i] += out.grad[i];
            if (broadcastRows) {
                for (let i = 0; i < r; i++)
                    for (let j = 0; j < c; j++) other.grad[j] += out.grad[this.idx(i, j)];
            } else {
                for (let i = 0; i < r * c; i++) other.grad[i] += out.grad[i];
            }
        };
        return out;
    }

    sub(other: Tensor): Tensor {
        return this.add(other.mulScalar(-1));
    }

    mulScalar(s: number): Tensor {
        const outData = this.data.map(v => v * s);
        const out = new Tensor(outData, this.shape, [this]);
        out._backward = () => {
            for (let i = 0; i < this.data.length; i++) this.grad[i] += s * out.grad[i];
        };
        return out;
    }

    relu(): Tensor {
        const outData = this.data.map(v => Math.max(0, v));
        const out = new Tensor(outData, this.shape, [this]);
        out._backward = () => {
            for (let i = 0; i < this.data.length; i++) {
                this.grad[i] += (out.data[i] > 0 ? 1 : 0) * out.grad[i];
            }
        };
        return out;
    }

    // Sum of (this ** 2) -- used directly as our loss function (sum squared error)
    sumSquares(): Tensor {
        let total = 0;
        for (const v of this.data) total += v * v;
        const out = new Tensor([total], [1, 1], [this]);
        out._backward = () => {
            for (let i = 0; i < this.data.length; i++) {
                this.grad[i] += 2 * this.data[i] * out.grad[0];
            }
        };
        return out;
    }

    backward(): void {
        const topo: Tensor[] = [];
        const visited = new Set<Tensor>();
        const build = (v: Tensor) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of (v as any)._prev) build(child);
                topo.push(v);
            }
        };
        build(this);
        this.grad.fill(0);
        this.grad[0] = 1;
        for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
    }

    zeroGrad(): void {
        this.grad.fill(0);
    }
}
