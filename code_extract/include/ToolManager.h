#pragma once
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/Support/Casting.h"

#include "CodeDB.h"

namespace clang {
namespace tooling {
class ClangTool;
}
class ASTUnit;
class FunctionDecl;
} // namespace clang

/// @brief Exposes API to pull a single function by name and manages other tool
/// aspects.
class ToolManager {
  clang::tooling::ClangTool const *tool;
  ObjInfo *primaryFn;
  std::vector<std::unique_ptr<clang::ASTUnit>> asts;
  std::unique_ptr<CodeDB> db;

public:
  explicit ToolManager(clang::tooling::ClangTool *tool);

  /// @brief Finds the function declaration by name.
  /// @param fnName Name of the function declaration to find.
  /// @return The function declaration if a function by that name exists in the
  /// tool's compilation database, null otherwise.
  ObjInfo *findFnDeclByName(std::string const &fnName);

  /// @brief Pulls the function definition for the specified function name and
  /// all its dependencies.
  /// @param fnName Name of the function to get dependencies for.
  void getStandaloneFnContext(std::string const &fnName);
};
