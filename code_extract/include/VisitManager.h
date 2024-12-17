#pragma once

#include <queue>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Casting.h"

#include "CodeDB.h"

namespace clang {
class NamedDecl;
class Decl;
class Expr;
class FunctionDecl;
class TagDecl;
class VarDecl;
class Stmt;
class ASTContext;
} // namespace clang

/// @brief Manages nodes to visit and decls to add to the extracted body.
class VisitManager {
  CodeDB const &db;
  ObjInfo &primaryFn;
  std::queue<clang::Stmt *> toVisitNodes;
  std::vector<clang::NamedDecl const *> declRefs;
  std::vector<clang::TagDecl const *> tagDecls;
  std::unordered_map<std::string, ObjInfo const *> visitedNodes;
  std::vector<std::string> includes;

public:
  VisitManager(ObjInfo &pf, CodeDB const &cdb) : db(cdb), primaryFn(pf) {}

  /// @brief Check if a named decl is visited or not.
  /// @param name Name (fully qualified) of the decl to check.
  bool isVisited(std::string const& name);

  /// @brief Gets the the visited objInfo by name.
  ObjInfo const* getVisitedObj(std::string const& name);

  /// @brief Marks a named decl visited.
  /// @param name Name (fully qualified) of the decl to mark.
  /// @param objInfo Object info associated with decl to mark.
  void markVisited(std::string name, ObjInfo const *objInfo);

  /// @brief Adds a clang statement to visit queue.
  void addToVisit(clang::Stmt *stmt);

  /// @brief Adds a global variable declaration to be emitted to code later.
  /// @param decl Global variable declaration.
  void registerDecl(clang::VarDecl const *decl);

  /// @brief Adds a function declaration to be emitted later. DO NOT call this
  /// for the primary function!
  /// @param decl Function declaration.
  void registerDecl(clang::FunctionDecl const *decl);

  /// @brief Adds the declaration of an enum/union/struct/class to be emitted
  /// later.
  /// @param decl Decl of the record/enum type.
  void registerDecl(clang::TagDecl const *decl);

  /// @brief Default instantiates all function parameters for the given
  /// functionDecl. For now it can be only called once as it writes out the
  /// instantiated parameters in code as 'p1, p2 ...'.
  /// @param fnDecl The function to instantiate params for.
  std::string getParamInstantiationsAsString(clang::FunctionDecl const *fnDecl);

  /// @brief Emits the standalone code file containing the searched function and
  /// all other dependencies into output.
  /// @param output String to dump the full standalone code into.
  void emitStandaloneFile(std::string &output);

  /// @brief Emits the standalone extracted code to filename and compiles it to objName.
  /// @param filename File name extracted code is written to.
  /// @param objName Name of the ouput compiled object.
  void emitAndCompilePrimaryFile(std::string const& filename,
                                 std::string const& objName = "out");

  /// @brief Given a function declaration, pulls all dependencies necessary to
  /// make the function compile standalone.
  /// @param fnDecl The function declaration to pull dependencies for.
  void pullPrimaryFnContext();
};