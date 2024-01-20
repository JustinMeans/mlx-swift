import Foundation

public class RMSNorm : Module {
    
    let weight: MLXArray
    let eps: Float
 
    public init(_ dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    
    public override func describeParameters(_ indent: Int) -> String {
        "(dimensions=\(weight.dim(0)), eps=\(self.eps))"
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // S is 1/sqrt(N) where N is the size of the features of x and is used
        // to compute a numerically more stable RMS of x by multiplying with S
        // first and summing.
        //
        // This way we prefer underflow over overflow which is controlled with
        // the parameter epsilon anyway.
        
        let S = 1 / MLXArray(x.dim(-1)) ** 0.5
        
        var n = (x * S).square().sum(axis: -1, keepDims: true)
        n = rsqrt(n + eps)
        
        return weight * x * n
    }
}

