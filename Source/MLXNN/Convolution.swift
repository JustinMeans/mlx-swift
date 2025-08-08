// Copyright © 2024 Apple Inc.

import Foundation
import MLX

private func circularPad2d(_ x: MLXArray, padding: (Int, Int)) -> MLXArray {
    let (_, H, W, _) = x.shape4
    let (padH, padW) = padding
    guard padH > 0 || padW > 0 else { return x }

    var y = x

    if padH > 0 {
        let top = x[0..., H - padH ..< H, 0..., 0...]
        let bottom = x[0..., 0 ..< padH, 0..., 0...]
        y = concatenated([top, y, bottom], axis: 1)
    }

    if padW > 0 {
        let left = y[0..., 0..., W - padW ..< W, 0...]
        let right = y[0..., 0..., 0 ..< padW, 0...]
        y = concatenated([left, y, right], axis: 2)
    }

    return y
}

/// Applies a 1-dimensional convolution over the multi-channel input sequence.
///
/// ### See Also
/// - ``Conv2d``
/// - ``Conv3d``
/// - ``init(inputChannels:outputChannels:kernelSize:stride:padding:dilation:groups:bias:)``
open class Conv1d: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: Int
    public let dilation: Int
    public let stride: Int
    public let groups: Int

    /// Applies a 1-dimensional convolution over the multi-channel input sequence.
    ///
    /// The channels are expected to be last i.e. the input shape should be `NLC` where:
    ///
    /// - `N` is the batch dimension
    /// - `L` is the sequence length
    /// - `C` is the number of input channels
    ///
    /// - Parameters:
    ///   - inputChannels: number of input channels (`C` from the discussion)
    ///   - outputChannels: number of output channels
    ///   - kernelSize: size of the convolution filters
    ///   - stride: stride when applying the filter
    ///   - padding: how many positions to 0-pad the input with
    ///   - dilation: dilation of the convolution
    ///   - groups: the number of groups for the convolution
    ///   - bias: if `true` add a learnable bias to the output
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))

        precondition(inputChannels % groups == 0, "Input channels must be divisible by groups")

        self.weight = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(
            x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

/// Applies a 2-dimensional convolution over the multi-channel input image.
///
/// ### See Also
/// - ``Conv1d``
/// - ``Conv3d``
/// - ``init(inputChannels:outputChannels:kernelSize:stride:padding:dilation:groups:bias:)``
public enum Conv2dPaddingMode {
    case zeros
    case circular
}

open class Conv2d: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: (Int, Int)
    public let dilation: (Int, Int)
    public let stride: (Int, Int)
    public let groups: Int
    public let paddingMode: Conv2dPaddingMode

    /// Applies a 2-dimensional convolution over the multi-channel input image.
    ///
    /// The channels are expected to be last i.e. the input shape should be `NHWC` where:
    ///
    /// - `N` is the batch dimension
    /// - `H` is the input image height
    /// - `W` is the input image width
    /// - `C` is the number of input channels
    ///
    /// - Parameters:
    ///   - inputChannels: number of input channels (`C` from the discussion)
    ///   - outputChannels: number of output channels
    ///   - kernelSize: size of the convolution filters
    ///   - stride: stride when applying the filter
    ///   - padding: how many positions to 0-pad the input with
    ///   - dilation: dilation of the convolution
    ///   - groups: the number of groups for the convolution
    ///   - bias: if `true` add a learnable bias to the output
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        groups: Int = 1,
        bias: Bool = true,
        paddingMode: Conv2dPaddingMode = .zeros
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize.first * kernelSize.second))

        precondition(inputChannels % groups == 0, "Input channels must be divisible by groups")

        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputChannels, kernelSize.first, kernelSize.second, inputChannels / groups])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding.values
        self.dilation = dilation.values
        self.stride = stride.values
        self.groups = groups
        self.paddingMode = paddingMode
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input: MLXArray
        switch paddingMode {
        case .zeros:
            input = x
        case .circular:
            input = circularPad2d(x, padding: padding)
        }

        var y = conv2d(
            input, weight, stride: .init(stride), padding: .init(paddingMode == .zeros ? padding : (0, 0)), dilation: .init(dilation),
            groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

/// Applies a 3-dimensional convolution over the multi-channel input image.
///
/// ### See Also
/// - ``Conv1d``
/// - ``Conv2d``
/// - ``init(inputChannels:outputChannels:kernelSize:stride:padding:dilation:groups:bias:)``
open class Conv3d: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: (Int, Int, Int)
    public let dilation: (Int, Int, Int)
    public let stride: (Int, Int, Int)
    public let groups: Int

    /// Applies a 3-dimensional convolution over the multi-channel input image.
    ///
    /// The channels are expected to be last i.e. the input shape should be `NDHWC` where:
    ///
    /// - `N` is the batch dimension
    /// - `D` is the input image depth
    /// - `H` is the input image height
    /// - `W` is the input image width
    /// - `C` is the number of input channels
    ///
    /// - Parameters:
    ///   - inputChannels: number of input channels (`C` from the discussion)
    ///   - outputChannels: number of output channels
    ///   - kernelSize: size of the convolution filters
    ///   - stride: stride when applying the filter
    ///   - padding: how many positions to 0-pad the input with
    ///   - dilation: dilation of the convolution
    ///   - groups: the number of groups for the convolution
    ///   - bias: if `true` add a learnable bias to the output
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrTriple,
        stride: IntOrTriple = 1,
        padding: IntOrTriple = 0,
        dilation: IntOrTriple = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(
            1 / Float(inputChannels * kernelSize.first * kernelSize.second * kernelSize.third))

        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputChannels, kernelSize.first, kernelSize.second, kernelSize.third, inputChannels])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding.values
        self.dilation = dilation.values
        self.stride = stride.values
        self.groups = groups
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv3d(
            x, weight, stride: .init(stride), padding: .init(padding), dilation: .init(dilation),
            groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}
