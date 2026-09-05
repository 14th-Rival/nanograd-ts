import { MathC } from "./MathC.ts";

export class Main {

    protected X: Float32Array = new Float32Array([1, 2, 3, 4, 5]);
    protected Y: Float32Array = new Float32Array([1, 4, 9, 16, 25]); // y = x^2

    protected nHidden: number = 4;
    protected W: Float32Array;
    protected B: Float32Array;
    protected V: Float32Array;
    protected c: number;
    protected lr: number = 0.002;

    public constructor() {
        this.W = Float32Array.from({ length: this.nHidden }, () => MathC.randomUniform(0.1, 1));
        this.B = Float32Array.from({ length: this.nHidden }, () => MathC.randomUniform(-1, 1));
        this.V = Float32Array.from({ length: this.nHidden }, () => MathC.randomUniform(-1, 1));
        this.c = MathC.randomUniform(-1, 1);

        for (let epoch = 1; epoch <= 5000; epoch++) {
            const n = this.X.length;

            // --- BATCHED FORWARD PASS ---
            // Instead of looping sample-by-sample, compute ALL samples x ALL neurons at once.
            // Z[sample][neuron] = X[sample] * W[neuron] + B[neuron]
            const Z: Float32Array[] = Array.from({ length: n }, (_, s) =>
                Float32Array.from({ length: this.nHidden }, (_, i) => this.X[s] * this.W[i] + this.B[i])
            );
            const H: Float32Array[] = Z.map(row => row.map(MathC.relu));
            const yPred: Float32Array = Float32Array.from(H.map(row =>
                row.reduce((sum, hi, i) => sum + hi * this.V[i], 0) + this.c
            ));
            const error: Float32Array = Float32Array.from(yPred.map((yp, s) => yp - this.Y[s]));

            let totalLoss = 0;
            for (const e of error) totalLoss += e * e;

            // --- BATCHED BACKWARD PASS ---
            // Accumulate gradients across ALL samples using array reduces instead of a manual accumulator loop
            const gW = new Float32Array(this.nHidden);
            const gB = new Float32Array(this.nHidden);
            const gV = new Float32Array(this.nHidden);
            let gC = 0;

            for (let s = 0; s < n; s++) {
                gC += 2 * error[s];
                for (let i = 0; i < this.nHidden; i++) {
                    gV[i] += 2 * error[s] * H[s][i];
                    const dz = 2 * error[s] * this.V[i] * MathC.reluDeriv(Z[s][i]);
                    gW[i] += dz * this.X[s];
                    gB[i] += dz;
                }
            }

            for (let i = 0; i < this.nHidden; i++) {
                this.W[i] -= this.lr * (gW[i] / n);
                this.B[i] -= this.lr * (gB[i] / n);
                this.V[i] -= this.lr * (gV[i] / n);
            }
            this.c -= this.lr * (gC / n);

            if (epoch % 500 === 0) {
                console.log(`Epoch ${epoch} | avg loss = ${totalLoss / n}`);
            }
        }

        for (const x of [1.5, 2.5, 3.5, 4.5, 6]) {
            console.log(`Prediction for x=${x}: ${this.predict(x)}  (true: ${x * x})`);
        }
    }

    protected predict(x: number): number {
        let sum = this.c;
        for (let i = 0; i < this.nHidden; i++) {
            sum += MathC.relu(x * this.W[i] + this.B[i]) * this.V[i];
        }
        return sum;
    }
}

new Main();