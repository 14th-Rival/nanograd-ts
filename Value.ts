// A tiny automatic differentiation engine, inspired by the same core idea
// behind PyTorch/TensorFlow's autograd. Every Value remembers how it was
// computed, so gradients can be found automatically via the chain rule.

export class Value {
    public data: number;
    public grad: number = 0;
    private _backward: () => void = () => {};
    private _prev: Set<Value>;

    constructor(data: number, children: Value[] = []) {
        this.data = data;
        this._prev = new Set(children);
    }

    static of(x: number | Value): Value {
        return x instanceof Value ? x : new Value(x);
    }

    add(other: number | Value): Value {
        const o = Value.of(other);
        const out = new Value(this.data + o.data, [this, o]);
        out._backward = () => {
            this.grad += out.grad;
            o.grad += out.grad;
        };
        return out;
    }

    mul(other: number | Value): Value {
        const o = Value.of(other);
        const out = new Value(this.data * o.data, [this, o]);
        out._backward = () => {
            this.grad += o.data * out.grad;
            o.grad += this.data * out.grad;
        };
        return out;
    }

    sub(other: number | Value): Value {
        return this.add(Value.of(other).mul(-1));
    }

    div(other: number | Value): Value {
        const o = Value.of(other);
        const out = new Value(this.data / o.data, [this, o]);
        out._backward = () => {
            this.grad += (1 / o.data) * out.grad;
            o.grad += (-this.data / (o.data ** 2)) * out.grad;
        };
        return out;
    }

    pow(n: number): Value {
        const out = new Value(this.data ** n, [this]);
        out._backward = () => {
            this.grad += n * (this.data ** (n - 1)) * out.grad;
        };
        return out;
    }

    relu(): Value {
        const out = new Value(Math.max(0, this.data), [this]);
        out._backward = () => {
            this.grad += (out.data > 0 ? 1 : 0) * out.grad;
        };
        return out;
    }

    // Walks the computation graph backward, applying the chain rule
    // at every step -- this replaces every hand-written gradient formula
    // we wrote in earlier versions of Main.ts.
    backward(): void {
        const topo: Value[] = [];
        const visited = new Set<Value>();

        const build = (v: Value) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._prev) build(child);
                topo.push(v);
            }
        };
        build(this);

        this.grad = 1;
        for (let i = topo.length - 1; i >= 0; i--) {
            topo[i]._backward();
        }
    }
}
