import { MathC } from "./MathC.ts";

export class Main {

    protected data: [number, number][] = [ [1, 2],  [2, 4], [3, 6], [4, 8], [5, 10] ];

    public constructor() {
        let w1 = MathC.randomUniform(-1, 1), b1 = MathC.randomUniform(-1, 1);
        let w2 = MathC.randomUniform(-1, 1), b2 = MathC.randomUniform(-1, 1);
        let v1 = MathC.randomUniform(-1, 1), v2 = MathC.randomUniform(-1, 1);
        let c = MathC.randomUniform(-1, 1);
        let lr = 0.01;

        for (let i = 1; i <= 2000; i++) {
            let total_loss = 0;
            let g_w1 = 0, g_b1 = 0, g_w2 = 0, g_b2 = 0, g_v1 = 0, g_v2 = 0, g_c = 0;

            for (let [x, y_true] of this.data) {
                // Forward pass — pre-activation (z) then activation (h)
                let z1 = w1 * x + b1;
                let z2 = w2 * x + b2;
                let h1 = MathC.relu(z1);
                let h2 = MathC.relu(z2);

                let y_pred = v1 * h1 + v2 * h2 + c;
                let error = y_pred - y_true;
                total_loss += error ** 2;

                // Output layer gradients (unchanged — one step from the error)
                g_v1 += 2 * error * h1;
                g_v2 += 2 * error * h2;
                g_c  += 2 * error;

                // Hidden layer gradients — chain rule now passes through ReLU's derivative
                let dL_dh1 = 2 * error * v1;
                let dL_dz1 = dL_dh1 * MathC.reluDeriv(z1);
                g_w1 += dL_dz1 * x;
                g_b1 += dL_dz1;

                let dL_dh2 = 2 * error * v2;
                let dL_dz2 = dL_dh2 * MathC.reluDeriv(z2);
                g_w2 += dL_dz2 * x;
                g_b2 += dL_dz2;
            }

            const n = this.data.length;
            w1 -= lr * (g_w1 / n); b1 -= lr * (g_b1 / n);
            w2 -= lr * (g_w2 / n); b2 -= lr * (g_b2 / n);
            v1 -= lr * (g_v1 / n); v2 -= lr * (g_v2 / n); c -= lr * (g_c / n);

            console.log(`Epoch ${i} | loss = ${total_loss / n}`);
        }

        const pred = v1 * MathC.relu(w1*10 + b1) + v2 * MathC.relu(w2*10 + b2) + c;
        console.log(`Prediction for x=10: ${pred}  (true: 20)`);
    }
}

new Main();