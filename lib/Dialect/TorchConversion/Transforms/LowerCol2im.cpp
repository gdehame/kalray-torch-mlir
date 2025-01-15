//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinTypes.h"
#include "PassDetail.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

namespace {
class LowerCol2imPass
    : public LowerCol2imBase<LowerCol2imPass> {

  // Lower col2im operations into a linalg operation computing the following:
  // for (int b = 0; b < nBatches; b++)
  // for (int c = 0; c < nChannels; c++)
  // for (int i = 0; i <= (paddedHeight - 1 - (kernelHeight - 1) * horizontalDilation) / horizontalStride; i++)
  // for (int j = 0; j <= (paddedWidth - 1 - (kernelWidth - 1) * verticalDilation) / verticalStride; j++)
  // for (int k = 0; k < kernelHeight; k++)
  // for (int l = 0; l < kernelWidth; l++)
  //   paddedOutput[b, c, i*verticalStride + k * verticalDilation, j * horizontalStride + l * horizontalDilation]
  //     = input[b, k * kernelWidth + l + c * kernelWidth * kernelHeight, 
  //               j + i * ((paddedWidth - 1 - (kernelWidth - 1) * horizontalDilation) / horizontalStride + 1)];
  // output = paddedOutput[:,:,verticalPadding:-verticalPadding, horizontalPadding:-horizontalPadding];

  void runOnOperation() override {
    mlir::FunctionOpInterface f = getOperation();
    f->walk([&](Torch::AtenCol2imOp col2imOp){
      // Retrieve the hyperparameters
      IRRewriter rewriter(col2imOp);
      Value input = col2imOp.getSelf();
      assert(col2imOp.getOutputSize().getDefiningOp && 
            isa<Torch::PrimListConstructOp>(col2imOp.getOutputSize().getDefiningOp()) && "Not implemented yet\n");
      Torch::PrimListConstructOp outputSizes = cast<Torch::PrimListConstructOp>(col2imOp.getOutputSize().getDefiningOp());
      assert(outputSizes.getNumOperands() == 2 && 
            outputSizes->getOperand(0).getDefiningOp() && 
            outputSizes->getOperand(1).getDefiningOp() &&
            isa<Torch::ConstantIntOp>(outputSizes->getOperand(0).getDefiningOp()) && "Not implemented yet\n");
      assert(isa<Torch::ConstantIntOp>(outputSizes->getOperand(1).getDefiningOp()) && "Not implemented yet\n");
      int height = cast<Torch::ConstantIntOp>(outputSizes->getOperand(0).getDefiningOp()).getValue();
      int width = cast<Torch::ConstantIntOp>(outputSizes->getOperand(1).getDefiningOp()).getValue();

      assert(col2imOp.getPadding().getDefiningOp && 
            isa<Torch::PrimListConstructOp>(col2imOp.getPadding().getDefiningOp()) && "Not implemented yet\n");
      Torch::PrimListConstructOp paddings = cast<Torch::PrimListConstructOp>(col2imOp.getPadding().getDefiningOp());
      assert(paddings.getNumOperands() == 2 && 
            paddings->getOperand(0).getDefiningOp() && 
            paddings->getOperand(1).getDefiningOp() &&
            isa<Torch::ConstantIntOp>(paddings->getOperand(0).getDefiningOp()) && "Not implemented yet\n");
      assert(isa<Torch::ConstantIntOp>(paddings->getOperand(1).getDefiningOp()) && "Not implemented yet\n");
      int horizontalPadding = cast<Torch::ConstantIntOp>(paddings->getOperand(1).getDefiningOp()).getValue();
      int verticalPadding = cast<Torch::ConstantIntOp>(paddings->getOperand(0).getDefiningOp()).getValue();

      int paddedWidth = width + 2 * horizontalPadding;
      int paddedHeight = height + 2 * verticalPadding;

      assert(col2imOp.getKernelSize().getDefiningOp && 
            isa<Torch::PrimListConstructOp>(col2imOp.getKernelSize().getDefiningOp()) && "Not implemented yet\n");
      Torch::PrimListConstructOp kerSizes = cast<Torch::PrimListConstructOp>(col2imOp.getKernelSize().getDefiningOp());
      assert(kerSizes.getNumOperands() == 2 && 
            kerSizes->getOperand(0).getDefiningOp() && 
            kerSizes->getOperand(1).getDefiningOp() &&
            isa<Torch::ConstantIntOp>(kerSizes->getOperand(0).getDefiningOp()) && "Not implemented yet\n");
      assert(isa<Torch::ConstantIntOp>(kerSizes->getOperand(1).getDefiningOp()) && "Not implemented yet\n");
      int kernelWidth = cast<Torch::ConstantIntOp>(kerSizes->getOperand(0).getDefiningOp()).getValue();
      int kernelHeight = cast<Torch::ConstantIntOp>(kerSizes->getOperand(1).getDefiningOp()).getValue();

      assert(col2imOp.getDilation().getDefiningOp && 
            isa<Torch::PrimListConstructOp>(col2imOp.getDilation().getDefiningOp()) && "Not implemented yet\n");
      Torch::PrimListConstructOp dilations = cast<Torch::PrimListConstructOp>(col2imOp.getDilation().getDefiningOp());
      assert(dilations.getNumOperands() == 2 && 
            dilations->getOperand(0).getDefiningOp() && 
            dilations->getOperand(1).getDefiningOp() &&
            isa<Torch::ConstantIntOp>(dilations->getOperand(0).getDefiningOp()) && "Not implemented yet\n");
      assert(isa<Torch::ConstantIntOp>(dilations->getOperand(1).getDefiningOp()) && "Not implemented yet\n");
      int verticalDilation = cast<Torch::ConstantIntOp>(dilations->getOperand(0).getDefiningOp()).getValue();
      int horizontalDilation = cast<Torch::ConstantIntOp>(dilations->getOperand(1).getDefiningOp()).getValue();

      assert(col2imOp.getStride().getDefiningOp && 
            isa<Torch::PrimListConstructOp>(col2imOp.getStride().getDefiningOp()) && "Not implemented yet\n");
      Torch::PrimListConstructOp strides = cast<Torch::PrimListConstructOp>(col2imOp.getStride().getDefiningOp());
      assert(strides.getNumOperands() == 2 && 
            strides->getOperand(0).getDefiningOp() && 
            strides->getOperand(1).getDefiningOp() &&
            isa<Torch::ConstantIntOp>(strides->getOperand(0).getDefiningOp()) && "Not implemented yet\n");
      assert(isa<Torch::ConstantIntOp>(strides->getOperand(1).getDefiningOp()) && "Not implemented yet\n");
      int verticalStride = cast<Torch::ConstantIntOp>(strides->getOperand(0).getDefiningOp()).getValue();
      int horizontalStride = cast<Torch::ConstantIntOp>(strides->getOperand(1).getDefiningOp()).getValue();
      
      // Create intermediate buffers
      TensorType outputType = cast<Torch::ValueTensorType>(col2imOp.getType()).toBuiltinTensor();

      Value outputBuffer = rewriter.create<tensor::EmptyOp>(col2imOp->getLoc(), 
                              ArrayRef<int64_t>{outputType.getDimSize(0), outputType.getDimSize(1), height, width}, 
                              outputType.getElementType());

      Value paddedOutput = rewriter.create<tensor::EmptyOp>(col2imOp->getLoc(), 
                            ArrayRef<int64_t>{outputType.getDimSize(0), outputType.getDimSize(1), paddedHeight, paddedWidth},
                            outputType.getElementType());

      // Create the linalg loop interators
      SmallVector<utils::IteratorType, 6> iteratorTypes(6, utils::IteratorType::reduction);
      iteratorTypes[0] = utils::IteratorType::parallel;
      iteratorTypes[1] = utils::IteratorType::parallel;
      
      SmallVector<AffineMap, 4> indexingMaps;
      AffineExpr b = rewriter.getAffineDimExpr(0);
      AffineExpr c = rewriter.getAffineDimExpr(1);
      AffineExpr i = rewriter.getAffineDimExpr(2);
      AffineExpr j = rewriter.getAffineDimExpr(3);
      AffineExpr k = rewriter.getAffineDimExpr(4);
      AffineExpr l = rewriter.getAffineDimExpr(5);

      indexingMaps.push_back(AffineMap::get(6, 0, 
                      ArrayRef<AffineExpr>{b, k*kernelWidth+l+c*kernelWidth*kernelHeight, 
                                          j+i*(1+(paddedWidth-1-(kernelWidth-1)*horizontalDilation) / horizontalStride)},
                      rewriter.getContext()));

      // We create 2 additional irrelevent indexing maps and inputs (kernel, upperBounds) so that the operation is able to find
      // the upper bounds of each loop. Otherwise we get the following error: "'linalg.generic' op expected the shape-to-loops map to be non-null"
      indexingMaps.push_back(AffineMap::get(6, 0, ArrayRef<AffineExpr>{k, l}, rewriter.getContext()));
      indexingMaps.push_back(AffineMap::get(6, 0, ArrayRef<AffineExpr>{i, j}, rewriter.getContext()));

      indexingMaps.push_back(AffineMap::get(6, 0, 
                      ArrayRef<AffineExpr>{b, c, 
                                                  i*verticalStride+k*verticalDilation, 
                                                  j*horizontalStride+l*horizontalDilation},
                      rewriter.getContext()));


      // The body of the linalg.generic op
      auto body = [&](OpBuilder& b, Location loc, ValueRange args) {
        Value acc = (outputType.getElementType().isInteger()) ?
                  b.create<arith::AddIOp>(loc, args[0], args[3]).getResult()
                  : (isa<FloatType>(outputType.getElementType()) ?
                    b.create<arith::AddFOp>(loc, args[0], args[3]).getResult()
                    : b.create<complex::AddOp>(loc, args[0], args[3]).getResult());
        b.create<linalg::YieldOp>(loc, acc);
      };

      input = rewriter.create<ToBuiltinTensorOp>(col2imOp->getLoc(), 
                                                cast<Torch::ValueTensorType>(input.getType()).toBuiltinTensor(), 
                                                input);
      
      // Create the "irrelevent" inputs
      Value kernel = rewriter.create<tensor::EmptyOp>(col2imOp->getLoc(),
                        ArrayRef<int64_t>{kernelWidth, kernelHeight},
                        outputType.getElementType());
      Value upperBounds = rewriter.create<tensor::EmptyOp>(col2imOp->getLoc(),
                        ArrayRef<int64_t>{1 + (paddedHeight - 1 - (kernelHeight - 1) * verticalDilation) / verticalStride, 
                                          1 + ((paddedWidth - 1 - (kernelWidth - 1) * horizontalDilation)) / horizontalStride},
                        outputType.getElementType());

      paddedOutput = rewriter.create<linalg::GenericOp>(col2imOp->getLoc(),
                                        paddedOutput.getType(),
                                        ValueRange{input, kernel, upperBounds},
                                        ValueRange(paddedOutput),
                                        indexingMaps, iteratorTypes, body)->getResult(0);
      
      // Remove the padding
      OpFoldResult one = rewriter.getI32IntegerAttr(1);
      OpFoldResult zero = rewriter.getI32IntegerAttr(0);
      OpFoldResult vpad = rewriter.getI32IntegerAttr(verticalPadding);
      OpFoldResult hpad = rewriter.getI32IntegerAttr(horizontalPadding);
      OpFoldResult vdim = rewriter.getI32IntegerAttr(height);
      OpFoldResult hdim = rewriter.getI32IntegerAttr(width);
      OpFoldResult batchSize = rewriter.getI32IntegerAttr(outputType.getDimSize(0));
      OpFoldResult nChannels = rewriter.getI32IntegerAttr(outputType.getDimSize(1));
      outputBuffer = rewriter.create<tensor::ExtractSliceOp>(col2imOp->getLoc(),
                                                            paddedOutput,
                                                            ArrayRef<Range>{
                                                              Range{zero, batchSize, one},
                                                              Range{zero, nChannels, one},
                                                              Range{vpad, vdim, one},
                                                              Range{hpad, hdim, one}
                                                            });
      rewriter.setInsertionPoint(col2imOp);
      FromBuiltinTensorOp newOp = rewriter.create<FromBuiltinTensorOp>(col2imOp->getLoc(), col2imOp.getType(), outputBuffer);
      rewriter.replaceAllUsesWith(col2imOp, newOp);
      rewriter.eraseOp(col2imOp);
    });
  }
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> 
mlir::torch::TorchConversion::createLowerCol2imPass() {
  return std::make_unique<LowerCol2imPass>();
}
