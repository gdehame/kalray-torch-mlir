//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "PassDetail.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/AsmState.h"
#include <string>
#include <fstream>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

namespace {
class ExternalizeDenseConstantPass
    : public ExternalizeDenseConstantBase<
          ExternalizeDenseConstantPass> {
  uint weightOffset = 0;
  std::string weightOutput;
  
  LogicalResult initialize(MLIRContext* ctx) override {
    //Initialize the pass by defining the path to the binary weights output file and clearing it if it already exists
    assert(workDir.length() > 0 && "No working directory specified to ExternalizeDenseConstant pass, please provide one through option \"work-dir\"");
    weightOutput = workDir + "/weights.bin";
    std::ofstream ofs(weightOutput, std::ios::trunc);
    ofs.close();
    return success();
  }

  void runOnOperation() override {
    assert(workDir.length() > 0 && "No working directory specified to ExternalizeDenseConstant pass, please provide one through option \"work-dir\"");
    
    //Iterate through the module to find every arith.constant defining dense objects
    ModuleOp m = getOperation();
    m.walk([&](func::FuncOp f) {
      f.walk([&](arith::ConstantOp op) {

        if (auto denseElem = dyn_cast<DenseElementsAttr>(op.getValueAttr())) {
          //For each dense constant: 
          //  - create a global uninitialized memref
          //  - replace the arith.constant with a load from that global
          //  - dump the dense constant in the binary weights output file (appending to previous weights)
          //  - compute and add as attribute to the two operations the offset in the binary file at which the dense constant starts 
          std::string name = "kvx_weight" + std::to_string(weightOffset);
          OpBuilder builder = OpBuilder(m->getContext());
          builder.setInsertionPointToStart(m.getBody());
          
          ShapedType shapedType = dyn_cast<ShapedType>(op.getType());
          if (!shapedType)
            return;

          memref::GlobalOp globalWeight = builder.create<memref::GlobalOp>(m->getLoc(), 
                                                                                  builder.getStringAttr(name),
                                                                                  builder.getStringAttr("private"),
                                                                                  MemRefType::get(shapedType.getShape(), shapedType.getElementType()),
                                                                                  UnitAttr::get(m->getContext()),
                                                                                  false,
                                                                                  nullptr);
          
          globalWeight->setAttr("weightOffset", builder.getIndexAttr(weightOffset));
          if (weightOffset == 0)
            globalWeight->setAttr("weightPath", builder.getStringAttr(weightOutput));

          builder.setInsertionPointAfter(op);
          memref::GetGlobalOp weight = builder.create<memref::GetGlobalOp>(op->getLoc(), 
                                                                          MemRefType::get(shapedType.getShape(), shapedType.getElementType()), 
                                                                          StringRef(name));
          builder.setInsertionPointAfter(weight);
          Value castedWeight = builder.create<bufferization::ToTensorOp>(op.getLoc(), op.getType(), weight, true).getResult();
          
          weight->setAttr("weightOffset", builder.getIndexAttr(weightOffset));
          if (weightOffset == 0)
            weight->setAttr("weightPath", builder.getStringAttr(weightOutput));

          op.replaceAllUsesWith(castedWeight);

          ArrayRef<char> rawData = denseElem.getRawData();
          std::ofstream ofs(weightOutput, std::ios::binary | std::ios::app);
          ofs.write(rawData.data(), rawData.size());
          weightOffset += rawData.size();
          ofs.flush();
          ofs.close();
        }

      });
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::torch::TorchConversion::createExternalizeDenseConstantPass() {
  return std::make_unique<ExternalizeDenseConstantPass>();
}
