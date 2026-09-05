import { Tensor } from './Tensor.ts';

export class MainTensor {

    // 5 samples, 1 feature each -- shape [5, 1]
    protected X: Tensor = new Tensor([1, 2, 3, 4, 5], [5, 1]);
    protected Y: Tensor = new Tensor([1, 4, 9, 16, 25], [5, 1]); // y = x^2

    protected nHidden: number = 4;
    protected W1: Tensor = Tensor.randomUniform([1, this.nHidden], 0.1, 0.9);
    protected B1: Tensor = Tensor.randomUniform([1, this.nHidden]);
    protected W2: Tensor = Tensor.randomUniform([this.nHidden, 1]);
    protected B2: Tensor = Tensor.randomUniform([1, 1]);
    protected lr: number = 0.002;

    public constructor() {
        for (let epoch = 1; epoch <= 5000; epoch++) {
            // --- ENTIRE BATCH, ONE FORWARD PASS, NO PER-SAMPLE LOOP ---
            const Z1 = this.X.matmul(this.W1).add(this.B1);  // [5,1] @ [1,4] -> [5,4]
            const H1 = Z1.relu();
            const Z2 = H1.matmul(this.W2).add(this.B2);       // [5,4] @ [4,1] -> [5,1]
            const loss = Z2.sub(this.Y).sumSquares();

            [this.W1, this.B1, this.W2, this.B2].forEach(p => p.zeroGrad());
            loss.backward();

            for (const p of [this.W1, this.B1, this.W2, this.B2]) {
                for (let i = 0; i < p.data.length; i++) {
                    p.data[i] -= this.lr * (p.grad[i] / this.X.shape[0]);
                }
            }

            if (epoch % 500 === 0) {
                console.log(`Epoch ${epoch} | loss = ${loss.data[0] / this.X.shape[0]}`);
            }
        }

        const testX = new Tensor([1.5, 2.5, 3.5, 4.5, 6], [5, 1]);
        const H1 = testX.matmul(this.W1).add(this.B1).relu();
        const preds = H1.matmul(this.W2).add(this.B2);
        for (let i = 0; i < testX.data.length; i++) {
            const x = testX.data[i];
            console.log(`Prediction for x=${x}: ${preds.data[i]}  (true: ${x * x})`);
        }
    }
}

new MainTensor();
