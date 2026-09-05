import { MathC } from "./MathC.ts";

export class Main {

    protected data: [number, number][] = [ [1, 2],  [2, 4], [3, 6], [4, 8], [5, 10] ];

    public constructor() {
        let weight: number = MathC.randomUniform(-1, 1);
        let bias: number = MathC.randomUniform(-1, 1);
        let learningRate: number = 0.01;

        for (let i = 1; i <= 2000; i++) {
            let total_loss: number = 0;
            let total_gradient_weight: number = 0;
            let total_gradient_bias: number = 0;

            for (let [x, y_true] of this.data) {
                let y_pred = weight * x + bias;
                let error = y_pred - y_true;
                let loss = error ** 2;

                total_loss += loss;

                let gradient_w = 2 * error * x;
                let gradient_b = 2 * error;

                total_gradient_weight += gradient_w;
                total_gradient_bias += gradient_b;
            }

            let avg_gradient_weight = total_gradient_weight / this.data.length;
            let avg_gradient_bias = total_gradient_bias / this.data.length;

            weight = weight - learningRate * avg_gradient_weight;
            bias = bias - learningRate * avg_gradient_bias;

            let avg_loss = total_loss / this.data.length;
            console.log(`Epoch ${i} | w = ${weight} | b = ${bias} | avg loss = ${avg_loss}`);
        }

        console.log(`Final: w = ${weight}, b = ${bias}  (true answer is w=2.0, b=0.0)`);
        console.log(`Model's guess for x=10: ${weight*10 + bias}  (true answer is 20)`);
    }
}

new Main();