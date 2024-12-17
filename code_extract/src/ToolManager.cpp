#include "ToolManager.h"
#include "CodeDB.h"
#include "Visitor.h"

#include <iostream>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"

namespace ct = clang::tooling;
namespace matcher = clang::ast_matchers;

ToolManager::ToolManager(ct::ClangTool *tool) : tool(tool) {
  tool->buildASTs(asts);
  db.reset(new CodeDB());
  // Build code database
  for (auto &ast : asts) {
    auto &ctx = ast->getASTContext();
    CodeExtractVisitor vis(*db.get(), ctx);
    // Should move this in a run function or ctor
    vis.TraverseAST(ctx);
  }
}

ObjInfo *ToolManager::findFnDeclByName(std::string const &fnName) {
  auto obj = db->getObjInfoOrNull(fnName);

  if (!obj) {
    std::cerr << "Could not find function(s) named " << fnName
              << " within specified project!" << std::endl;
    return nullptr;
  }
  if (!obj->getDefiniton()) {
    std::cerr << "Could not find body for function(s) named " << fnName
              << " within specified project! Make sure it is not an externally "
                 "defined function!"
              << std::endl;
    return nullptr;
  }

  return obj;
}

void ToolManager::getStandaloneFnContext(std::string const &fnName) {
  primaryFn = findFnDeclByName(fnName);
  VisitManager mv(*primaryFn, *db.get());

  mv.pullPrimaryFnContext();
  std::string out;
  mv.emitStandaloneFile(out);
  mv.emitAndCompilePrimaryFile(fnName + ".cpp");
  /// TODO: Complete
}