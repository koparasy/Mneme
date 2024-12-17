#include "VisitManager.h"
#include "Visitor.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/raw_ostream.h"

namespace ct = clang::tooling;
namespace matcher = clang::ast_matchers;

void VisitManager::registerDecl(clang::VarDecl const *decl) {
  declRefs.push_back(decl);
}

void VisitManager::registerDecl(clang::TagDecl const *decl) {
  tagDecls.push_back(decl);
}

void VisitManager::registerDecl(clang::FunctionDecl const *decl) {
  // Do not emit inlined functions
  if (decl->isCXXClassMember() && decl->isInlined())
    return;
  declRefs.push_back(decl);
}

bool VisitManager::isVisited(std::string const &name) {
  return visitedNodes.find(name) != visitedNodes.end();
}

ObjInfo const *VisitManager::getVisitedObj(std::string const &name) {
  return visitedNodes.at(name);
}

void VisitManager::markVisited(std::string name, ObjInfo const *objInfo) {
  visitedNodes.insert({name, objInfo});
}

void VisitManager::addToVisit(clang::Stmt *stmt) { toVisitNodes.push(stmt); }

std::string VisitManager::getParamInstantiationsAsString(
    clang::FunctionDecl const *fnDecl) {
  auto numParams = fnDecl->getNumParams();
  if (!numParams)
    return "";

  // Typically we would want to find all function call to this method
  // but for now we can just use types from functionDecl
  std::string paramInst;
  std::stringstream paramInstStream(paramInst);
  // Right now we only build the declarations, eventually we should be able to
  // recreate the values...
  /// TODO: this
  for (int i = 0; i < numParams; i++) {
    auto expr = fnDecl->parameters()[i];
    std::string typeString = expr->getType().getAsString();
    paramInstStream << typeString << " p" << i + 1 << ";\n";
  }

  return paramInstStream.str();
}

void VisitManager::emitStandaloneFile(std::string &output) {
  llvm::raw_string_ostream ss(output);

  for (auto &inc : includes) {
    ss << "#include \"" << inc << "\"\n";
  }

  for (auto &tags : tagDecls) {
    // Same with tags, add missing semicolon!
    tags->print(ss);
    ss << ";\n";
  }

  // Emit all declrefs (functions calls + global refs)
  for (auto ref_it = declRefs.rbegin(); ref_it != declRefs.rend(); ref_it++) {
    auto decl = *ref_it;
    decl->print(ss);
    if (decl->getKind() == clang::Decl::Kind::Var)
      ss << ";";
    ss << "\n";
  }

  auto body =
      static_cast<clang::FunctionDecl const *>(primaryFn.getDefiniton());
  // Building main
  ss << "int main(int argc, char *argv[]) {\n"
     << getParamInstantiationsAsString(body) << body->getNameAsString();

  // Build function call
  auto numParams = body->getNumParams();
  int paramCount = 1;
  ss << "(";
  for (; paramCount < numParams; paramCount++)
    ss << "p" << paramCount << ", ";

  if (paramCount <= numParams)
    ss << "p" << paramCount;
  ss << ");\n";
  ss << "}\n";
}

void VisitManager::emitAndCompilePrimaryFile(std::string const &filename,
                                             std::string const &objName) {
  // First dump everything to a file
  std::ofstream outFile(filename);

  if (!outFile) {
    std::cerr << "Error creating file " << filename << std::endl;
    return;
  }
  std::string code;

  emitStandaloneFile(code);

  outFile << code;
  outFile.close();

  // Then clang-format
  std::string command = "clang-format -i " + filename;
  if (system(command.c_str()) != 0)
    std::cerr << "Formatting failed!" << std::endl;

  // Then compile
  command = "clang++ -o " + objName + " " + filename;
  if (system(command.c_str()) != 0) {
    std::cerr << "Compilation failed!" << std::endl;
  } else {
    std::cout << "Compilation successful!" << std::endl;
  }

  /// FIXME: We need to add a lot more here....should not be restrited to a
  /// specific compiler + include paths, link libs etc.
}

void VisitManager::pullPrimaryFnContext() {
  auto primaryDecl = primaryFn.getDefiniton()->getAsFunction();
  addToVisit(primaryDecl->getBody());
  registerDecl(primaryDecl);
  markVisited(primaryFn.getName(), &primaryFn);

  MatchVisitor mv(*this, db);
  mv.VisitParms(primaryDecl);
  while (!toVisitNodes.empty()) {
    auto stmt = toVisitNodes.front();
    toVisitNodes.pop();
    mv.TraverseStmt(stmt);
  }
  /// TODO: Complete
}