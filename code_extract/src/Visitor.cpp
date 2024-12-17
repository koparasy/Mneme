#include "Visitor.h"

#include "clang/AST/Decl.h"

#include <iostream>

namespace helper {
template <typename T>
void storeDecl(T *decl, CodeDB &cdb, std::string name = "") {
  // Make use of compile-time polymorphism
  if (name.empty())
    name = decl->getQualifiedNameAsString();
  clang::Decl *defDecl = decl->getDefinition();
  if (!cdb.isRegistered(name)) {
    cdb.registerDecl(name, decl, defDecl);
  } else {
    if (defDecl)
      cdb.addDefinitionDecl(name, defDecl);
  }
}

template <typename T>
T const *visitAndRegister(clang::NamedDecl const *decl, VisitManager &vm,
                          CodeDB const &cdb) {
  std::string name = decl->getQualifiedNameAsString();
  if (vm.isVisited(name))
    return static_cast<T const*>(vm.getVisitedObj(name)->getDefiniton());

  assert(cdb.isRegistered(name) && "All decl to visit should be registered!");
  auto objInfo = cdb.getObjInfoOrNull(name);

  // Lookup definition from database
  auto defDecl = static_cast<T const *>(objInfo->getDefiniton());
  vm.markVisited(name, objInfo);
  vm.registerDecl(defDecl);

  return defDecl;
}

/// @brief Checks if the given input is either of the types specifed in the
/// variadic template parameters.
/// @tparam A Types to check against.
/// @tparam ...B Types to check against.
/// @param expr Expression to check for.
/// @return True if the input expression is of any one type specifed in the
/// template parameter. False otherwise.
template <class A, class... B> bool isOneOf(clang::Expr const *expr) {
  if constexpr (sizeof...(B) == 0)
    return false;
  else {
    if (llvm::dyn_cast<A>(expr))
      return true;
    else
      return isOneOf<B...>(expr);
  }
}

bool isGlobalVar(clang::VarDecl const *decl) {
  return !decl->isStaticLocal() && decl->hasGlobalStorage();
}

clang::Type const *getUnderlyingCanonicalType(clang::QualType const &type) {
  auto canonType = type.getTypePtr();
  while (canonType->isPointerType() || canonType->isArrayType()) {
    canonType = canonType->getPointeeOrArrayElementType();
  }
  return canonType->getUnqualifiedDesugaredType();
}

clang::RecordDecl *getAsRecordType(clang::QualType qt) {
  return getUnderlyingCanonicalType(qt)->getAsRecordDecl();
}

bool handleVarDecl(clang::QualType qt, VisitManager &vm, CodeDB const &codedb) {
  auto recordDecl = helper::getAsRecordType(qt);
  if (!recordDecl)
    return true;

  // We will typically not find RecordDecls within function bodies or init
  // expressions Hence, we need to visit them when we encounter either their var
  // decl or static function call (unsupported as of yet).
  helper::visitAndRegister<clang::RecordDecl>(recordDecl, vm, codedb);

  // Also we do not want to visit anything else from here as we will visit the
  // function calls separately
  return true;
}
} // namespace helper

bool CodeExtractVisitor::VisitVarDecl(clang::VarDecl *decl) {
  if (!helper::isGlobalVar(decl))
    return true;
  helper::storeDecl(decl, codedb);
  return true;
}

bool CodeExtractVisitor::VisitFunctionDecl(clang::FunctionDecl *decl) {
  helper::storeDecl(decl, codedb);
  return true;
}

bool CodeExtractVisitor::VisitRecordDecl(clang::RecordDecl *decl) {
  helper::storeDecl(decl, codedb);
  return true;
}

bool MatchVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
  auto ctor = expr->getConstructor();
  helper::visitAndRegister<clang::RecordDecl>(ctor->getParent(), vm, codedb);
  return true;
}

bool MatchVisitor::VisitDeclRefExpr(clang::DeclRefExpr *declRef) {
  // Need to filter out global varDeclRef...
  auto varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl());
  if (!varDecl || !helper::isGlobalVar(varDecl))
    return true;

  auto defDecl = helper::visitAndRegister<clang::VarDecl>(varDecl, vm, codedb);
  assert(
      defDecl->hasDefinition() &&
      "We should have seen this variable's decl before unless it is external!");

  /// FIXIT: Replace with better design...
  auto varInit = const_cast<clang::VarDecl *>(defDecl)->getInit();
  // If the init expression is not a literal, we should visit it to resolve
  // dependencies. Obviously this list of literals is not exhaustive.
  if (varInit &&
      !helper::isOneOf<clang::CXXBoolLiteralExpr, clang::CharacterLiteral,
                       clang::FixedPointLiteral, clang::FloatingLiteral,
                       clang::IntegerLiteral, clang::StringLiteral>(varInit))
    vm.addToVisit(varInit);

  // For globals that are themselves record types (for e.g., enums), we should
  // visit their decls
  return helper::handleVarDecl(varDecl->getType(), vm, codedb);
}

bool MatchVisitor::VisitCallExpr(clang::CallExpr *callExpr) {
  auto decl = callExpr->getDirectCallee();
  assert(decl && "As of now we only support function decls as callees!");
  auto defDecl =
      helper::visitAndRegister<clang::FunctionDecl>(decl, vm, codedb);
  if (!defDecl->hasBody())
    return true;
  vm.addToVisit(defDecl->getBody());

  // Handle function param var decls
  VisitParms(defDecl);

  if (defDecl->isCXXClassMember() && decl->isStatic()) {
    auto recordDecl = static_cast<clang::CXXMethodDecl *>(decl)->getParent();
    helper::visitAndRegister<clang::RecordDecl>(recordDecl, vm, codedb);
  }
  return true;
}

void MatchVisitor::VisitParms(clang::FunctionDecl const *defDecl) {
  for (auto param_it = defDecl->param_begin(); param_it != defDecl->param_end();
       param_it++)
    helper::handleVarDecl((*param_it)->getType(), vm, codedb);
}