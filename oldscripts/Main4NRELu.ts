import { MathC } from "./MathC.ts";

export class Main {

    protected data: [number, number][] = [ [1, 1], [2, 4], [3, 9], [4, 16], [5, 25] ]; // y = x^2

    protected nHidden: number = 4;
    protected w: number[] = [];
    protected b: number[] = [];
    protected v: number[] = [];
    protected c: number = MathC.randomUniform(-1, 1);
    protected lr: number = 0.002;

    public constructor() {
        for (let i = 0; i < this.nHidden; i++) {
            this.w.push(MathC.randomUniform(0.1, 1)); // forced positive to avoid dying ReLU
            this.b.push(MathC.randomUniform(-1, 1));
            this.v.push(MathC.randomUniform(-1, 1));
        }

        for (let epoch = 1; epoch <= 5000; epoch++) {
            let total_loss = 0;
            let g_w = new Array(this.nHidden).fill(0);
            let g_b = new Array(this.nHidden).fill(0);
            let g_v = new Array(this.nHidden).fill(0);
            let g_c = 0;

            for (let [x, y_true] of this.data) {
                let z = this.w.map((wi, i) => wi * x + this.b[i]);
                let h = z.map(MathC.relu);
                let y_pred = h.reduce((sum, hi, i) => sum + this.v[i] * hi, 0) + this.c;
                let error = y_pred - y_true;
                total_loss += error ** 2;

                g_c += 2 * error;
                for (let i = 0; i < this.nHidden; i++) {
                    g_v[i] += 2 * error * h[i];
                    let dz = 2 * error * this.v[i] * MathC.reluDeriv(z[i]);
                    g_w[i] += dz * x;
                    g_b[i] += dz;
                }
            }

            const n = this.data.length;
            for (let i = 0; i < this.nHidden; i++) {
                this.w[i] -= this.lr * (g_w[i] / n);
                this.b[i] -= this.lr * (g_b[i] / n);
                this.v[i] -= this.lr * (g_v[i] / n);
            }
            this.c -= this.lr * (g_c / n);

            if (epoch % 500 === 0) {
                console.log(`Epoch ${epoch} | avg loss = ${total_loss / n}`);
            }
        }

        // test on unseen x values to check generalization
        for (const x of [1.5, 2.5, 3.5, 4.5, 6]) {
            console.log(`Prediction for x=${x}: ${this.predict(x)}  (true: ${x*x})`);
        }
    }

    protected predict(x: number): number {
        const h = this.w.map((wi, i) => MathC.relu(wi * x + this.b[i]));
        return h.reduce((sum, hi, i) => sum + this.v[i] * hi, 0) + this.c;
    }
}

new Main();