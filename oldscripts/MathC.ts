export class MathC {
    
    public static randomUniform(min: number, max: number): number {
        return Math.random() * (max - min) + min;
    }
    public static relu(z: number): number {
        return Math.max(0, z);
    }
    public static reluDeriv(z: number): number {
        return z > 0 ? 1 : 0;
    }

}