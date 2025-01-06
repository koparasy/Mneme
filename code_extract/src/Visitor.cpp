#include "Visitor.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/ASTUnit.h"

namespace helper {
void addIncludeIfExternal(std::string const &name, clang::SourceLocation sloc,
                          clang::ASTContext const &ctx, CodeDB &codedb) {
  auto declFile = sloc.printToString(ctx.getSourceManager());
  declFile = declFile.substr(0, declFile.find_first_of(':'));

  /// FIXME: For now we use a trick to figure out if the include is system-wide
  /// or local. Typically local includes will show up as relative paths in the
  /// code's source location. Eventually we should make this check more robust
  /// by checking against the current project path.
  if (declFile[0] == '/' && declFile.find(codedb.projPath) == std::string::npos)
    codedb.addExtSource(name, declFile);
}

bool checkPotentialInclude(clang::NamedDecl const *decl, VisitManager &vm,
                           CodeDB const &codedb) {
  auto extFileName = codedb.getExtSource(decl->getQualifiedNameAsString());
  return vm.registerInclude(extFileName);
}

template <typename T>
void storeDecl(T *decl, clang::ASTUnit const &unit, CodeDB &cdb,
               std::string name = "") {
  // Make use of compile-time polymorphism
  if (name.empty())
    name = decl->getQualifiedNameAsString();
  clang::Decl *defDecl = decl->getDefinition();
  if (!cdb.isRegistered(name)) {
    cdb.registerDecl(unit, name, decl, defDecl);
  } else {
    if (defDecl)
      cdb.addDefinitionDecl(name, defDecl);
  }
  addIncludeIfExternal(name, decl->getSourceRange().getBegin(),
                       unit.getASTContext(), cdb);
}

template <typename T>
std::tuple<T const *, bool> visitAndRegister(clang::NamedDecl const *decl,
                                             VisitManager &vm,
                                             CodeDB const &cdb) {
  std::string name = decl->getQualifiedNameAsString();
  if (vm.isVisited(name))
    return {static_cast<T const *>(vm.getVisitedObj(name)->getDefiniton()),
            false};

  if (!cdb.isRegistered(name))
    decl->dump();
  assert(cdb.isRegistered(name) && "All decl to visit should be registered!");
  auto objInfo = cdb.getObjInfoOrNull(name);

  // Lookup definition from database
  auto defDecl = static_cast<T const *>(objInfo->getDefiniton());
  vm.markVisited(name, objInfo);
  vm.registerDecl(defDecl);

  return {defDecl, true};
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
  return canonType;
}

clang::RecordDecl *getAsRecordType(clang::QualType qt) {
  return getUnderlyingCanonicalType(qt)->getAsRecordDecl();
}

// Use the fact that builtin functions are typically prepended with "__"
bool isPotentialBuiltinByName(std::string const &name) {
  return name.size() > 2 && '_' == name[0] && '_' == name[1];
}

void handleRecordDecl(clang::RecordDecl const *recordDecl, VisitManager &vm,
                      CodeDB const &codedb) {
  // If externally defined (or built-in), do not include def as we will include
  // the file itself.
  if (checkPotentialInclude(recordDecl, vm, codedb) ||
      isPotentialBuiltinByName(recordDecl->getNameAsString()))
    return;

  // We will typically not find RecordDecls within function bodies or init
  // expressions. Hence, we need to visit them when we encounter either their
  // var decl or static function call (unsupported as of yet).
  helper::visitAndRegister<clang::RecordDecl>(recordDecl, vm, codedb);

  // Also we do not want to visit anything else from here as we will visit the
  // function calls separately
}

// A little bit of code duplication for clarity.
void handleTypedefs(clang::TypedefType const *typ, VisitManager &vm,
                    CodeDB const &codedb) {
  if (!typ)
    return;

  auto typDecl = typ->getDecl();
  // If externally defined (or built-in), do not include def as we will
  // include the file itself.
  if (checkPotentialInclude(typDecl, vm, codedb) ||
      isPotentialBuiltinByName(typDecl->getNameAsString()))
    return;

  handleTypedefs(typDecl->getUnderlyingType()->getAs<clang::TypedefType>(), vm,
                 codedb);

  visitAndRegister<clang::TypedefNameDecl>(typDecl, vm, codedb);
}

void handleVarDecl(clang::QualType qt, VisitManager &vm, CodeDB const &codedb) {
  auto cannonType = getUnderlyingCanonicalType(qt);
  clang::RecordDecl const *decl = helper::getAsRecordType(qt);
  if (!decl) {
    auto typedefType = cannonType->getAs<clang::TypedefType>();
    // recursively visit underlying type(defs).
    // We should hopefully
    // create a handle typedef function that does not revisit typedef chains
    // again if visited before.
    handleTypedefs(typedefType, vm, codedb);
  } else
    handleRecordDecl(decl, vm, codedb);
}
} // namespace helper

bool CodeExtractVisitor::VisitVarDecl(clang::VarDecl *decl) {
  if (!helper::isGlobalVar(decl))
    return true;
  helper::storeDecl(decl, unit, codedb);
  return true;
}

bool CodeExtractVisitor::VisitFunctionDecl(clang::FunctionDecl *decl) {
  helper::storeDecl(decl, unit, codedb);
  return true;
}

bool CodeExtractVisitor::VisitRecordDecl(clang::RecordDecl *decl) {
  helper::storeDecl(decl, unit, codedb);
  return true;
}

/// FIXME: We do not need to cache these as typically defs are together with
/// decls.
bool CodeExtractVisitor::VisitTypedefNameDecl(clang::TypedefNameDecl *decl) {
  auto name = decl->getQualifiedNameAsString();
  if (codedb.isRegistered(name))
    return true;
  codedb.registerDecl(unit, name, decl, decl);
  helper::addIncludeIfExternal(name, decl->getSourceRange().getBegin(),
                               unit.getASTContext(), codedb);
  return true;
}

bool MatchVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
  auto ctor = expr->getConstructor();
  if (helper::isPotentialBuiltinByName(ctor->getNameAsString()))
    return true;
  helper::handleRecordDecl(ctor->getParent(), vm, codedb);
  return true;
}

bool MatchVisitor::VisitDeclRefExpr(clang::DeclRefExpr *declRef) {
  // Need to filter out global varDeclRef...
  auto varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl());
  if (!varDecl)
    return true;

  // Even if the vardecls are not interesting, visit their type.
  helper::handleVarDecl(declRef->getType(), vm, codedb);

  // Need to filter out cuda internals, for now we check if either the variable
  // name or variable type name is potentially builtin. OR if the variable decl
  // itself has a builtin attribute.
  bool isPotentialBuiltin =
      helper::isPotentialBuiltinByName(varDecl->getNameAsString()) ||
      helper::isPotentialBuiltinByName(
          varDecl->getType().getUnqualifiedType().getAsString()) ||
      varDecl->hasAttr<clang::BuiltinAttr>();
  if (!helper::isGlobalVar(varDecl) || isPotentialBuiltin)
    return true;

  auto [defDecl, visitBody] =
      helper::visitAndRegister<clang::VarDecl>(varDecl, vm, codedb);
  if (!visitBody)
    return true;

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

  return true;
}

bool MatchVisitor::VisitCallExpr(clang::CallExpr *callExpr) {
  auto decl = callExpr->getDirectCallee();
  assert(decl && "As of now we only support function decls as callees!");

  if (helper::checkPotentialInclude(decl, vm, codedb) ||
      helper::isPotentialBuiltinByName(decl->getNameAsString()))
    return true;

  auto [defDecl, visitBody] =
      helper::visitAndRegister<clang::FunctionDecl>(decl, vm, codedb);
  if (!visitBody || !defDecl->hasBody())
    return true;
  vm.addToVisit(defDecl->getBody());

  // Handle function param var decls
  VisitParms(defDecl);

  if (defDecl->isCXXClassMember() && decl->isStatic()) {
    auto recordDecl = static_cast<clang::CXXMethodDecl *>(decl)->getParent();
    helper::handleRecordDecl(recordDecl, vm, codedb);
  }
  return true;
}

void MatchVisitor::VisitParms(clang::FunctionDecl const *defDecl) {
  for (auto param_it = defDecl->param_begin(); param_it != defDecl->param_end();
       param_it++)
    helper::handleVarDecl((*param_it)->getType(), vm, codedb);
}